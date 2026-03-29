# AI Talent Lab 문의하기 Agent PoC 보고서

> **현재 브랜치: `indiv_nolabel`**
>
> | 항목 | 내용 |
> |------|------|
> | RAG 방식 | similarity-only (label 무시, 순수 코사인 유사도) |
> | 개인화 DB | user_db.py (수강생 SQLite) |
> | Tool 실행 | tools.py (Group 3: CODE_REVIEW_RESET, LITERACY_PRACTICE_RESET) |
> | History 라벨 | pre_label=False (휴리스틱 라벨 미부여) |

---

## PoC 목표

1. 답변이 가능한 질문과 그렇지 않은 질문은 어떤 것이 있는지?
2. 특정 context가 있어야 답할 수 있는 질문이 있다면, 그 context는 무엇인지?
3. 과거 응답을 바탕으로 일관성 있게 참고해서 답변하게 만들 수 있는지?

---

## 1. 데이터 현황

| 구분 | 문의 | 댓글 | 비고 |
|------|-----:|-----:|------|
| 전체 (train+test) | 110건 | 125건 | inquiry_all.json + inquiry_comment_all.json (기존+3월 신규 병합) |
| 테스트 (평가용) | 10건 | — | inquiry_all.json에서 random 샘플링 (random_state=42) |
| RAG 풀 (학습용) | 100건 | — | 테스트 10건 제외 후 FAISS 인덱싱 (leakage 방지) |

### 문의 유형 분포 (Train 100건 기준)

| 유형 | 건수 | Agent 처리 방향 |
|------|-----:|----------------|
| 인증/버튼 비활성화 | 24건 | no_response (계정 직접 조치) |
| 강의/과제/실습 관련 | 35건 | tool_rag 또는 human_review |
| 기술 에러 (코드/API) | 15건 | tool_rag |
| 접근/접속 문제 | 5건 | no_response (플랫폼 이슈) |
| 기타 / 미분류 | 5건 | no_response |

---

## 2. 처리 흐름

### 전체 흐름도

```
문의 입력 (title + content, HTML → 텍스트 변환)
    │
    ▼
[Step 0] 개인화 컨텍스트 조회
    │  user_db.py (SQLite): 수강생 수강 이력·기수·수료 상태 조회
    │  → LLM 분류·답변 프롬프트에 개인 컨텍스트로 주입
    │
    ▼
[Step 1] LLM 분류  (Azure gpt-5.2)
    │  ↑ prior_knowledge (플랫폼 사전 지식) 주입 — knowledge_base.json
    │  ↑ 라벨별 설명 + 실제 예시 제목 주입 — knowledge_base.json
    │  ↑ 기수 일정·인증시험 일정 주입 — schedule.json (동적, 기수마다 갱신)
    │
    │  label       → 12개 카테고리 중 하나 (복합 시 우선순위 높은 라벨)
    │  confidence  → high / low  (①문의 명확성 ②카테고리 단일성만으로 판단)
    │  is_compound → 성격이 다른 질문이 2개 이상 포함 여부
    │  sub_labels  → 감지된 모든 라벨 목록
    │
    ▼
[Step 2] 코드가 1차 strategy 결정
    │
    ├─ [복합 문의] is_compound=true AND sub_labels 中 Group1 라벨 포함
    │      → human_review  (RAG로 답변 가능한 부분 초안 + 운영자가 나머지 처리)
    │
    ├─ [복합 문의] is_compound=true AND sub_labels 全 Group2 AND 2–3개
    │      → tool_rag  (sub_label 각각 RAG 검색 후 context 합산 → 통합 답변 생성)
    │           └─ min(RAG score) < 0.65 → human_review 1차 다운그레이드
    │
    ├─ [복합 문의] is_compound=true AND sub_labels 全 Group2 AND 4개 이상
    │      → human_review  (복잡도 초과, 운영자 검토)
    │
    ├─ Group 1 라벨 OR confidence == low
    │      → no_response   (운영자 에스컬레이션, 즉시 종료)
    │
    └─ confidence == high  (Group 2)
           → RAG 검색 후 max_score 확인
                │
                ├─ max_score < 0.65  → human_review 1차 다운그레이드
                └─ max_score ≥ 0.65  → tool_rag (잠정)

[Step 3] RAG 검색  (_build_kb_context, similarity_only=True)
    FAISS 유사도 검색 (Azure text-embedding-3-large 3072차원)

    [similarity-only 모드]
    ① label 무시, 순수 코사인 유사도 상위 top_k(=5) 반환
    ② history 문서는 pre_label=False (휴리스틱 라벨 미부여)

    ③ CODE_LOGIC_ERROR 또는 에러 키워드 → error_solutions regex 보완
    ④ COURSE_INFO 라벨 → prior_knowledge.programs 과정 정보 추가
    → max_score 반환 (Step 2 1차 다운그레이드 판단에 사용)

[Step 4] 답변 생성 + LLM 자체 평가  (Azure gpt-5.2)
    실제 KB context를 모두 본 상태에서 답변 생성
    → answer           : 답변 텍스트
    → answer_confidence: high / medium / low  (KB 근거 기반 자체 평가)
    → uncertain_parts  : 확신 없는 부분 설명

[Step 5] 2차 strategy 분기 (답변 신뢰도 기반, 다운그레이드만)
    ├─ answer_confidence == low    → no_response  (KB 근거 부족, 에스컬레이션)
    ├─ answer_confidence == medium → human_review ([초안] 태그 부착, 운영자 검토)
    └─ answer_confidence == high   → 기존 strategy 유지 (tool_rag이면 자동 게시)
```

---

## 3. 카테고리 Labels — 12개 (Group 1: 5, Group 2: 5, Group 3: 2)

### Group 1 — `no_response` (운영자 에스컬레이션)

| Label | 해당 문의 | 실제 데이터 예시 |
|-------|----------|----------------|
| `ACCOUNT_ACTION_REQUIRED` | 개인 계정·권한·인증 직접 조치 필요 | "인증시험 버튼이 비활성화 되어있습니다.", "[AI Literacy] 인증 시작하기 버튼 비활성화", "AI Literacy 실습하기 문의(인증시험버튼 활성화)", "[긴급] 미국법인 인증 시험 활성화 필요 (1/29 오후 4시)" |
| `PLATFORM_SYSTEM_ERROR` | 플랫폼 서버·시스템 에러 | "console 접근/python script 실행이 안됩니다.", "[Boot Camp] 무한 로딩이 되고 있습니다 - 파이널 과제 제출", "최종과제 접속불가", "streamlit 서비스 개발 강의 중 실습 실행이 안됩니다." |
| `VIDEO_PLAYBACK_ERROR` | 강의 영상 재생 안됨 | "동영상 강의가 짤린 것 같아요", "강의 이어 듣기 재생이 않되네요" |
| `FEATURE_REQUEST` | 기능 개선·건의 | "최종 과제 영역에 '3개의 강의를 완료하면 과제를 시작할 수 있습니다'를 '전체 강의'로 변경 요청", "실습 파일 다운로드할 수 있게 공유 부탁 드립니다.", "최종 결과발표 전 사전으로 검토나 피드백을 받고싶습니다." |
| `UNCATEGORIZED` | 내용 불명확, 여러 주제 혼재 | "1", "삭제" |

### Group 2 — `tool_rag` (RAG 기반 답변 시도)

| Label | 해당 문의 | 실제 데이터 예시 |
|-------|----------|----------------|
| `COURSE_INFO` | 강의 목록, 수강 방법, 커리큘럼, 수료 조건 | "3개의 강의를 완료하면 과제를 시작할 수 있다고 했는데 안되요", "AI Bootcamp 수료 시, AI Literacy도 수료한 것으로 처리되나요?", "BootCamp 강의 복습 불가하나요?", "ai bootcamp 어디서 신청하면 되나요?" |
| `SUBMISSION_POLICY` | 과제 제출 횟수, 마감, 재제출 규정, 결과 발표 | "[Boot Camp] 최종 과제 제출 관련 문의 - 2단계(개발 명세서) 저장 이후 수정 가능 여부", "[Bootcamp] 9기 최종과제 평가결과", "과제 제출 완료 시점 문의", "과제가 제출이 되어버렸습니다.", "부트캠프 수료 문의 드립니다." |
| `SERVICE_GUIDE` | 플랫폼 이용 방법, 가이드 요청 | "IDE 사용법 가이드 부탁드립니다.", "Bootcamp 최종과제 기획 및 설계 최초 양식 문의", "새로 오픈한 코드리뷰 사용 문의", "실습 환경에 파이썬 패키지 설치 문의" |
| `ASSIGNMENT_DEVELOPMENT` | 과제 구현 방법, 개발 방향, 아키텍처 | "Azure OpenAI & LangChain 과제 개발 문의", "[Boot Camp] Sub-Graph 구조 관련 문의", "AI Bootcamp 최종과제 주제 문의드립니다.", "Bootcamp 최종과제 api key 관련 문의" |
| `CODE_LOGIC_ERROR` | 코드 에러, API 호출·파싱·rate limit 오류 | "[Boot Camp] RAG 강의 Generation - Retriever 관련 질문", "sqlite문의", "지금 인베딩 동시 요청 초과 인가요?", "에러 해결이 안돼요", "강의 중 소스코드 문의" |

### Group 3 — `tool_action` (에이전트 직접 실행)

| Label | 해당 문의 | Tool 종류 | 실행 내용 |
|-------|----------|----------|----------|
| `CODE_REVIEW_RESET` | 코드 리뷰 횟수 초기화 요청 | AUTO_TOOL | `user_db.reset_review_count()` — 당일 사용 횟수 0으로 초기화, 하루 10회 재사용 가능 |
| `LITERACY_PRACTICE_RESET` | AI Literacy 사전연습 횟수 복구 요청 | AUTO_TOOL | `user_db.restore_practice_count()` — 잘못 차감된 연습 횟수 복구 (기본 1회) |

> AUTO_TOOL은 즉시 실행(저위험, 단순 DB 조작). APPROVAL_TOOL은 관리자 승인 후 실행 예정(추후 추가).
> Group 3 라벨은 `ACCOUNT_ACTION_REQUIRED`와 혼동 주의 — 에이전트가 직접 처리 가능하므로 Group 3으로 분류.

---

## 4. 신뢰도(Confidence) — 2단계 분리 구조

### Step 1 — 분류 신뢰도 (텍스트만으로 판단, RAG 전)

| 번호 | 요소 | 판단 질문 |
|------|------|----------|
| ① | 문의 명확성 | 무엇을 묻는지 텍스트만 봐도 알 수 있는가? |
| ② | 카테고리 단일성 | 12개 중 딱 하나에만 해당하는가? (복합 문의로 sub_label 명확히 식별 가능한 경우는 ② 충족 간주) |

| 레벨 | 충족 조건 | 처리 방식 |
|------|----------|----------|
| `high` | ①② 모두 충족 | `tool_rag` 시도 (RAG 점수·답변 신뢰도로 추가 분기) |
| `low` | ① 또는 ② 미충족 | `no_response` → 에스컬레이션 (즉시 종료) |

> ③(답변 가능성)을 Step 1에서 제거한 이유: RAG 결과 없이 LLM이 추측할 수밖에 없어 부정확했음. 실제 KB context를 본 후 Step 4에서 자체 평가하는 것이 더 정확함.

### Step 4 — 답변 신뢰도 (LLM 자체 평가, RAG context 확인 후)

| 레벨 | 판단 기준 | 처리 방식 |
|------|----------|----------|
| `high` | 참고 정보에 명확한 근거가 있어 답변이 완결됨 | 기존 strategy 유지 (tool_rag이면 자동 게시) |
| `medium` | 참고 정보로 부분 답변 가능하나 운영자 확인 필요 | `human_review` → [초안] 태그 + 운영자 검토 |
| `low` | 참고 정보에 근거가 부족해 추측성 답변이 됨 | `no_response` → 에스컬레이션 |

> 답변에 "KB", "참고 정보", "시스템" 등 내부 용어 노출 금지. 수강생에게 보이는 답변에는 자연스러운 표현만 사용.

### 혼동 잦은 라벨 쌍 구분 기준 (LLM 프롬프트에 명시)

| 혼동 쌍 | 구분 기준 |
|---------|----------|
| `ACCOUNT_ACTION_REQUIRED` vs `PLATFORM_SYSTEM_ERROR` | 특정 사용자 계정·버튼 활성화를 운영자가 직접 바꿔야 → ACCOUNT. 시스템 자체 버그·장애 → PLATFORM |
| `SUBMISSION_POLICY` vs `COURSE_INFO` | 제출 횟수·마감·평가 발표 → SUBMISSION. 강의 이수 조건·커리큘럼·수료 관계 → COURSE |
| `CODE_LOGIC_ERROR` vs `ASSIGNMENT_DEVELOPMENT` | 에러 메시지·API 오류 → CODE. 설계 방향·아키텍처·구현 접근법 → ASSIGNMENT |

---

## 5. LLM vs 코드 역할 분리

**LLM이 판단하는 것**

| 단계 | 항목 | 내용 |
|------|------|------|
| Step 1 | `label` | 12개 카테고리 중 하나 (복합 시 우선순위 높은 라벨) |
| Step 1 | `confidence_level` | high / low (①문의 명확성 ②카테고리 단일성) |
| Step 1 | `is_compound` | 성격이 다른 질문이 2개 이상 포함 여부 |
| Step 1 | `sub_labels` | 감지된 모든 라벨 목록 |
| Step 4 | `answer_confidence` | high / medium / low (실제 KB context 보고 자체 평가) |
| Step 4 | `uncertain_parts` | 확신 없는 부분 설명 |

**코드가 결정하는 것**
```python
# Step 2: 1차 strategy 결정 (우선순위 순)
label.value in GROUP3_LABELS                               → tool_action    # Group3: 에이전트 직접 실행 (즉시 종료)
is_compound AND any(sub_label in GROUP1)                   → human_review   # Group1 포함 복합: 운영자 처리 필요
is_compound AND all(sub_label in GROUP2) AND 2<=count<=3   → tool_rag       # Group2 복합 2-3개: multi-RAG 합산
is_compound AND all(sub_label in GROUP2) AND count >= 4    → human_review   # Group2 복합 4개+: 복잡도 초과
label in Group1 OR confidence == 'low'                     → no_response    # 운영자 에스컬레이션 (즉시 종료)
confidence == 'high'                                       → tool_rag       # RAG 시도

# Step 3 (tool_action 경로): tools.py 실행기 위임 후 즉시 반환
# execute_tool_action(label_value, author_id, inquiry_text, user_db)
# → AUTO_TOOL  : 즉시 DB 조작 + 답변 생성 (CODE_REVIEW_RESET, LITERACY_PRACTICE_RESET)
# → APPROVAL_TOOL: 승인 대기 (추후 추가)

# Step 3 (RAG 경로): RAG 검색 — similarity_only=True (label 무시, 순수 유사도)
# 단일 문의:    kb_context, score = _build_kb_context(label, text, similarity_only=True)
# 복합 G2 문의: ctx_list = [_build_kb_context(lbl, text, similarity_only=True) for lbl in sub_labels]
#               kb_context = join(ctx_list),  score = min(scores)

# RAG 유사도 기반 1차 다운그레이드 (tool_rag만 해당)
strategy == 'tool_rag' AND score < 0.65
    → strategy='human_review'

# Step 5: 답변 신뢰도 기반 2차 분기 (다운그레이드만, tool_action 경로는 해당 없음)
answer_confidence == 'low'    → strategy='no_response'   # KB 근거 부족, 에스컬레이션
answer_confidence == 'medium' → strategy='human_review'  # 부분 답변 가능, 운영자 검토
answer_confidence == 'high'   → 기존 strategy 유지       # 자동 게시

# [초안] 태그: 최종 strategy가 human_review인 경우 부착
is_draft = (strategy == 'human_review')
```

---

## 6. 사전 지식 프롬프트 (Prior Knowledge)

`knowledge_base.json`의 `prior_knowledge` + `schedule.json`의 일정 정보를 **분류 프롬프트**와 **답변 생성 프롬프트** 양쪽에 동일하게 주입.

### Static — knowledge_base.json (거의 안 바뀌는 정보)

| 항목 | 내용 |
|------|------|
| 플랫폼 소개 | AI Talent Lab: SK AX그룹 임직원 대상 AI 교육 플랫폼 |
| 과정 구성 | AI Literacy (LV1, **상시 오픈**, VOD 6.5h) / AI Bootcamp (LV2, 6개 모듈, **기수제**, 4주 온라인) / AI Master Project (LV3) |
| 최종과제 규정 | 강의 6개 완료 후 시작, AI Agent 제작, 기획서+소스코드 각각 여러 번 제출, 마지막 제출 코드 기준 평가 |
| 인증 버튼 | 응시 대상자 + 응시 기간 중에만 활성화 → 비활성이 정상인 경우 있음 |
| IDE 정책 | 웹 기반 IDE 제공, 직접 venv 생성 비권장 (속도 저하·로딩 문제) |
| 수료 관계 | AI Bootcamp 수료 시 AI Literacy 동시 수료 처리 |
| 재수강 | 이전 기수 수강 이력과 무관하게 새 기수 재참여 가능 |

### Dynamic — schedule.json (기수마다 갱신)

| 항목 | 내용 |
|------|------|
| Bootcamp 현재 기수 | 12기 수강 기간, 과제 제출 마감, 결과 발표 예정일 |
| Bootcamp 예정 기수 | 3차·4차 신청 기간·수강 기간·모집 상태 |
| AI Literacy | 상시 운영 기간 (2026-01-01~12-31) |
| 인증시험 | 7차 시험일 (2026-03-18), 신청 기간, 응시 방법 |

### 효과

- 분류 시: "인증 버튼 비활성화 = 정상일 수 있음"을 LLM이 알고 ACCOUNT_ACTION_REQUIRED로 정확 분류
- 답변 시: 일정 관련 질문(다음 기수 언제, 신청 가능 여부)에 schedule.json으로 정확 답변
- 기수 변경 시 schedule.json만 수정 → knowledge_base.json 변경 불필요

---

## 7. Similarity-only RAG 설계

### 벡터 검색 스택

| 항목 | 내용 |
|------|------|
| 임베딩 모델 | Azure `text-embedding-3-large` (3072차원, endpoint 01) |
| 인덱스 | FAISS `IndexFlatIP` (코사인 유사도, L2 정규화 후 inner product) |
| 캐시 | `embeddings_cache.pkl` — 재실행 시 API 미호출 |
| 검색 방식 | **similarity-only**: label 무시, 순수 코사인 유사도 상위 top_k 반환 |
| 다운그레이드 임계값 | `RAG_CONFIDENCE_THRESHOLD = 0.65` — 미만이면 tool_rag → human_review |

### RAG 검색 흐름 (similarity-only 모드)

```
FAISS 검색 (순수 코사인 유사도 순)
    │
    └─ label 무관하게 유사도 상위 top_k(=5)개 반환
    │
    ▼ max_score 추출

+ 에러 솔루션 regex 보완  (CODE_LOGIC_ERROR 또는 에러 키워드 감지 시)
+ 과정 정보 보완           (COURSE_INFO 라벨 시)
```

### FAISS 인덱스 구성 문서

| 출처 | 문서 수 | label 여부 |
|------|--------:|-----------|
| KB 큐레이션 Q&A (knowledge_base.json) | ~20건 | 항상 있음 (검색 시 미사용) |
| 에러 솔루션 (knowledge_base.json) | 5건 | CODE_LOGIC_ERROR 고정 (검색 시 미사용) |
| history (inquiry_all.json 기반, 운영자 답변 있는 것) | ~205건 | pre_label=False (라벨 미부여, label=None) |
| **합계** | **~230건** | |

> `pre_label=False`: history 로드 시 휴리스틱 라벨을 부여하지 않음. similarity-only 모드에서는 label을 검색 필터로 사용하지 않으므로 라벨 없이 순수 유사도로 검색.

---

## 8. 필요 Context 정의

### Tier 1 — 핵심 Knowledge Base (현재 구현)

| Context | 내용 | 사용 Label |
|---------|------|-----------|
| 사전 지식 (prior_knowledge) | 플랫폼·과정·규정 핵심 사실 — knowledge_base.json (인증시험 실습횟수, 이수이력 갱신 정책 포함) | 전체 (분류+답변 프롬프트 공통 주입) |
| 운영 일정 (schedule) | 기수 일정·인증시험 일정 — schedule.json (동적) | 전체 (분류+답변 프롬프트 공통 주입) |
| FAISS 벡터 검색 | KB 큐레이션 Q&A + 에러 솔루션 + history (~240건) — Azure text-embedding-3-large (3072차원)<br>※ inquiry_all.json과 겹치는 qa_examples 제거 (leakage 방지) | 해당 Group 2 label |
| 에러 솔루션 정규식 | API 키, 패키지, 코드 에러 해결법 | `CODE_LOGIC_ERROR` 보완 |

### Tier 2 — 개인화 DB (user_db.py 구현)

| Context | 내용 | 구현 방식 |
|---------|------|----------|
| 수강생 수강 이력 | 등록 과정, 기수, 수료/진행중/미수료 상태 | `user_db.py` SQLite (`users`, `cohorts`, `enrollments` 테이블) |
| 개인화 컨텍스트 주입 | 수강생별 맞춤 컨텍스트 → LLM 분류·답변 프롬프트에 주입 | `UserContextDB.build_personal_context_str()` |
| 코드 리뷰 사용 현황 | 당일 코드 리뷰 사용 횟수·리셋 이력 | `user_db.py` SQLite (`code_review_logs` 테이블) — `CODE_REVIEW_RESET` Tool 실행 |
| AI Literacy 연습 횟수 | 누적 부여·사용·복구 이력 | `user_db.py` SQLite (`practice_sessions` 테이블) — `LITERACY_PRACTICE_RESET` Tool 실행 |

> 실서비스 전환 시 PostgreSQL 등으로 교체 가능 (SQLAlchemy 기반으로 변경하면 됨)

### Tier 3 — 실시간 조회 (미구현)

| Context | 내용 | 필요 이유 |
|---------|------|----------|
| 학습 진행률·인증 응시 자격 | 실시간 플랫폼 DB 조회 | 더 정밀한 개인화 답변 |
| 과제 제출 이력 | 제출 횟수, 마감 여부 | `SUBMISSION_POLICY` 정밀 답변 |

### Tier 4 — 고도화 (선택)

| Context | 내용 | 비고 |
|---------|------|------|
| 이미지 분석 | 에러 스크린샷 → 에러 메시지 추출 | Vision API 비용 증가 |

---

## 9. 다국어 지원

- **현재 데이터**: Train/Test 모두 100% 한국어
- **구현 상태**: 언어 자동 감지 (한국어 / 영어 / 일본어) + 언어별 인사말 분기 완료
- **검증 필요**: 영어·일본어 실제 문의 데이터 수집 후 테스트

---

## 10. 구현 파일 목록

| 파일 | 설명 |
|------|------|
| `inquiry_agent.py` | 메인 Agent (분류·RAG·답변 생성·strategy 결정, similarity_only=True) |
| `user_db.py` | 수강생 개인화 DB (SQLite) + Tool용 코드리뷰·연습횟수 관리 |
| `tools.py` | Group 3 Tool 정의 및 실행기 (AUTO_TOOL / APPROVAL_TOOL) |
| `knowledge_base.json` | 사전 지식 + 큐레이션 Q&A + 에러 솔루션 (static) |
| `schedule.json` | 기수별 수강 일정 + 인증시험 일정 (dynamic, 기수마다 갱신) |
| `inquiry_all.json` | 전체 문의 게시물 (110건, 기존+3월 신규 병합) — train+test 통합 소스 |
| `inquiry_comment_all.json` | 전체 문의 댓글 (125건) |
| `pipeline_viz.html` | 전체 파이프라인 시각화 (브라우저에서 열기) |
| `embeddings_cache.pkl` | 임베딩 캐시 (재실행 시 API 미호출) |
| `requirements.txt` | `openai>=1.0.0`, `faiss-cpu==1.7.4`, `numpy>=1.24.0,<2` |

---

## 11. 다음 단계

| Phase | 내용 |
|-------|------|
| ~~Phase 2~~ | ~~Vector DB 구축~~ → **완료** (FAISS + Azure text-embedding-3-large 3072차원, embeddings_cache.pkl 캐시) |
| **Phase 3** | DB 연동 — 사용자 수강 정보·과제 이력 실시간 조회 |
| **Phase 4** | Vision API 연동 — 에러 스크린샷 자동 분석 |
| **Phase 5** | 운영자 대시보드 — `human_review` 초안 검토·게시 UI |
