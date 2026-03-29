# AI Talent Lab 문의하기 Agent PoC

AI Talent Lab 문의하기 게시판에 자동 응답 Agent를 적용하기 위한 PoC.

> **현재 브랜치: `indiv_nolabel`**
> | 항목 | 내용 |
> |------|------|
> | RAG 방식 | similarity-only (label 무시, 순수 코사인 유사도) |
> | 개인화 DB | user_db.py (수강생 SQLite) |
> | Tool 실행 | tools.py (Group 3: CODE_REVIEW_RESET, LITERACY_PRACTICE_RESET) |
> | History 라벨 | pre_label=False (휴리스틱 라벨 미부여) |

---

## 파일 구조

```
70793/
├── inquiry_agent.py              # Agent 핵심 구현 (실행 파일)
├── user_db.py                    # 수강생 개인화 DB (SQLite)
├── tools.py                      # Group 3 Tool 정의 및 실행기 (AUTO_TOOL / APPROVAL_TOOL)
├── test_personalized.json        # 개인화 테스트 데이터
├── knowledge_base.json           # 라벨별 큐레이션 Q&A + 사전 지식 (static, 거의 안 바뀜)
├── schedule.json                 # 기수별 일정 + 인증시험 일정 (dynamic, 기수마다 갱신)
│
├── inquiry_all.json              # 전체 문의 게시물 (110건, 기존+3월 신규 병합)
├── inquiry_comment_all.json      # 전체 문의 댓글 (125건)
│                                 # ※ 테스트: inquiry_all.json에서 random 10건 샘플링 (random_state=42)
│                                 #   나머지 100건은 RAG 풀로 사용 (leakage 방지)
│
├── files.json                    # 첨부파일 메타데이터
├── files/                        # 실제 첨부파일
├── README.md                     # 본 파일
└── PoC_Report.md                 # PoC 상세 결과 (흐름 설계 + 분석)
```

> 운영자 ID: 2, 7, 61, 442, 2425, 3417

---

## 데이터 구성

| 구분 | 파일 | 문의 | 댓글 | 비고 |
|------|------|-----:|-----:|------|
| 전체 (train+test) | inquiry_all.json + inquiry_comment_all.json | 110건 | 125건 | 기존 + 3월 신규 병합 |
| 테스트 (평가용) | inquiry_all.json 내 random 샘플링 | 10건 | — | random_state=42, leakage 방지 |
| RAG 풀 (학습용) | inquiry_all.json 내 나머지 | 100건 | — | 테스트 10건 제외 후 인덱싱 |

---

## 실행 방법

### 1. 환경 준비

```bash
pip install openai faiss-cpu==1.7.4 "numpy<2"
```

`.env` 파일에 Azure OpenAI 설정:

```
# LLM - Azure OpenAI 02 (gpt-5.2)
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource-02>.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=gpt-5.2

# Embedding - Azure OpenAI 01 (text-embedding-3-large)
AZURE_OPENAI_EMBED_API_KEY=<key>
AZURE_OPENAI_EMBED_ENDPOINT=https://<resource-01>.openai.azure.com/
AZURE_EMBED_DEPLOYMENT=text-embedding-3-large
```

### 2. Agent 실행

```bash
cd 70793
python inquiry_agent.py
```

### 3. 출력 예시

```
테스트 1: 과제 제출 문의
────────────────────────────────────────────────────────────
[Label]       SUBMISSION_POLICY
[신뢰도]      high
[Strategy]    RAG 자동 답변 게시
[판단 근거]   과제 재제출 규정에 대한 명확한 문의
[답변 신뢰도] high

[생성된 답변]
안녕하세요, AI Talent Lab입니다.
최종과제의 기획 설계 문서와 소스코드는 여러 번 제출 가능합니다.
감사합니다.
```

---

## 처리 흐름 요약

```
문의 입력
  ↓
[Step 0] 개인화 컨텍스트 조회 (user_db.py)
  ↑ 수강생 수강 이력·기수·수료 상태 → LLM 프롬프트에 주입
  ↓
[Step 1] LLM 분류  →  label(12개) + confidence_level(high/low) + is_compound + sub_labels
          ↑ prior_knowledge + 라벨별 설명·예시 (knowledge_base.json) 주입
          ↑ 기수 일정·인증시험 일정 (schedule.json) 주입
          ↑ 수강생 개인 컨텍스트 주입 (user_db.py)
          ※ confidence는 ①문의 명확성 ②카테고리 단일성만으로 판단 (RAG 없이 판단 가능한 것)
  ↓
[Step 2] 코드가 1차 strategy 결정
  ├─ Group 3 라벨 (CODE_REVIEW_RESET, LITERACY_PRACTICE_RESET)
  │      →  tool_action  (에이전트가 DB 직접 조작 후 즉시 답변, 즉시 종료)
  ├─ 복합 문의 (is_compound) + sub_labels 中 Group1 포함
  │      →  human_review  (RAG 초안 + 운영자가 나머지 처리)
  ├─ 복합 문의 (is_compound) + sub_labels 全 Group2 + 2–3개
  │      →  tool_rag  (sub_label 각각 RAG 후 context 합산 → 통합 답변)
  ├─ 복합 문의 (is_compound) + sub_labels 全 Group2 + 4개 이상
  │      →  human_review  (복잡도 초과, 운영자 검토)
  ├─ Group 1 라벨 OR confidence == low  →  no_response  (운영자 에스컬레이션, 즉시 종료)
  └─ confidence == high                 →  tool_rag (RAG 점수로 다운그레이드 가능)
  ↓
[Step 3 — tool_action 경로] tools.py 실행기 위임
  - AUTO_TOOL  : 즉시 실행 (저위험, 단순 DB 조작)
    · CODE_REVIEW_RESET       → user_db.reset_review_count()   (당일 사용 횟수 0 초기화)
    · LITERACY_PRACTICE_RESET → user_db.restore_practice_count() (연습 횟수 복구)
  - APPROVAL_TOOL: 관리자 승인 후 실행 (추후 추가)
  ↓ (tool_rag / human_review 경로는 아래로)
[Step 3 — RAG 경로] RAG 검색 (tool_rag / human_review)
  ① FAISS 벡터 검색 (Azure text-embedding-3-large, 3072차원)
     - similarity-only: label 무시, 순수 코사인 유사도 top-5 반환
     - history 문서는 pre_label=False (휴리스틱 라벨 미부여)
     ※ 복합 문의 Group2 2-3개: sub_label별 각각 검색 후 context 합산, min(score)로 다운그레이드 판단
  ② 에러 솔루션 정규식 보완 (CODE_LOGIC_ERROR 또는 에러 키워드 감지 시)
  ③ 과정 정보 보완 (COURSE_INFO)
  ※ RAG score < 0.65 → tool_rag을 human_review로 1차 다운그레이드
  ↓
[Step 4] 답변 생성 + LLM 자체 평가 (Azure gpt-5.2)
  → 실제 KB context를 보고 답변 생성
  → answer_confidence (high/medium/low) 자체 평가 — "내가 충분한 근거로 답변하고 있는가"
  → uncertain_parts: 확신 없는 부분 설명
  ↓
[Step 5] 2차 strategy 분기 (답변 신뢰도 기반, 다운그레이드만)
  ├─ answer_confidence == low    →  no_response  (에스컬레이션)
  ├─ answer_confidence == medium →  human_review ([초안] 태그 부착)
  └─ answer_confidence == high   →  기존 strategy 유지 (자동 게시)
  ※ embeddings_cache.pkl 에 임베딩 결과 캐시 → 재실행 시 API 미호출
```

상세 내용은 [PoC_Report.md](PoC_Report.md) 참고.
