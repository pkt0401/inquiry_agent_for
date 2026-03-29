# platform.uname() hangs on this Windows server — patch before any openai import
import platform as _platform
_platform.system = lambda: "Windows"
_platform.platform = lambda: "Windows-11"
_platform.machine = lambda: "AMD64"

"""
AI Talent Lab 문의하기 Agent PoC

흐름:
  문의 입력
  → [Step 1] LLM 분류: label(10개) + confidence_level(high/low)
  → [Step 2] 코드가 strategy 결정:
       Group 1 라벨 또는 confidence==low → no_response  (운영자 에스컬레이션)
       confidence==high + Group2         → tool_rag     (RAG 시도, 유사도 낮으면 human_review)

RAG:
  - knowledge_base.json 에서 해당 label의 큐레이션 예제 우선 검색
  - 그 외 history(inquiry_all.json + inquiry_comment_all.json)에서 유사 Q&A 보조 검색
  - 사전 지식 (prior_knowledge) 은 분류·답변 양쪽 프롬프트에 주입
"""

import json
import re
import os
import random
import hashlib
import pickle
import numpy as np
faiss = None  # lazy import — build_index()에서 로드 (초기 기동 속도 개선)
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import AzureOpenAI
from dotenv import load_dotenv
from user_db import UserContextDB, CODE_REVIEW_DAILY_LIMIT
from tools import GROUP3_LABELS, get_tool_type, execute_tool_action
load_dotenv()
# ──────────────────────────────────────────────────────────────────
# HTML 전처리
# ──────────────────────────────────────────────────────────────────

class _HTMLTextExtractor(HTMLParser):
    _NEWLINE_TAGS = {'p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                     'li', 'tr', 'pre', 'blockquote'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._NEWLINE_TAGS:
            self._parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self._NEWLINE_TAGS:
            self._parts.append('\n')

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        text = ''.join(self._parts)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()


def html_to_text(html: str) -> str:
    if not html:
        return ''
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


# ──────────────────────────────────────────────────────────────────
# 카테고리 Labels — 10개
# ──────────────────────────────────────────────────────────────────

class InquiryLabel(str, Enum):
    # Group 1 — no_response (운영자 에스컬레이션)
    ACCOUNT_ACTION_REQUIRED = "ACCOUNT_ACTION_REQUIRED"   # 개인 계정·권한·인증 직접 조치 필요
    PLATFORM_SYSTEM_ERROR   = "PLATFORM_SYSTEM_ERROR"     # 플랫폼 서버·시스템 에러
    VIDEO_PLAYBACK_ERROR    = "VIDEO_PLAYBACK_ERROR"      # 강의 영상 재생 안됨
    FEATURE_REQUEST         = "FEATURE_REQUEST"           # 기능 개선·건의
    UNCATEGORIZED           = "UNCATEGORIZED"             # 내용 불명확·분류 불가

    # Group 2 — tool_rag (RAG 기반 답변 시도)
    COURSE_INFO             = "COURSE_INFO"               # 강의 목록·수강 방법·커리큘럼
    SUBMISSION_POLICY       = "SUBMISSION_POLICY"         # 과제 제출 횟수·마감·재제출 규정
    SERVICE_GUIDE           = "SERVICE_GUIDE"             # 플랫폼 이용 방법·가이드
    ASSIGNMENT_DEVELOPMENT  = "ASSIGNMENT_DEVELOPMENT"    # 과제 구현 방법·개발 방향
    CODE_LOGIC_ERROR        = "CODE_LOGIC_ERROR"          # 코드 에러·API 호출·파싱 오류

    # Group 3 — tool_action (에이전트 직접 실행)
    CODE_REVIEW_RESET       = "CODE_REVIEW_RESET"         # 코드 리뷰 횟수 초기화 요청
    LITERACY_PRACTICE_RESET = "LITERACY_PRACTICE_RESET"   # AI Literacy 사전연습 횟수 복구 요청


GROUP1 = {
    InquiryLabel.ACCOUNT_ACTION_REQUIRED,
    InquiryLabel.PLATFORM_SYSTEM_ERROR,
    InquiryLabel.VIDEO_PLAYBACK_ERROR,
    InquiryLabel.FEATURE_REQUEST,
    InquiryLabel.UNCATEGORIZED,
}

GROUP2 = {
    InquiryLabel.COURSE_INFO,
    InquiryLabel.SUBMISSION_POLICY,
    InquiryLabel.SERVICE_GUIDE,
    InquiryLabel.ASSIGNMENT_DEVELOPMENT,
    InquiryLabel.CODE_LOGIC_ERROR,
}


# RAG 유사도 임계값: 이 값 미만이면 tool_rag → human_review 다운그레이드
RAG_CONFIDENCE_THRESHOLD = 0.65


# ──────────────────────────────────────────────────────────────────
# 신뢰도 / Strategy
# ──────────────────────────────────────────────────────────────────

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    LOW  = "low"


class Strategy(str, Enum):
    NO_RESPONSE   = "no_response"
    HUMAN_REVIEW  = "human_review"
    TOOL_RAG      = "tool_rag"
    TOOL_ACTION   = "tool_action"   # 에이전트 직접 실행 (DB 조작 등)


# ──────────────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────────────

@dataclass
class LLMClassification:
    label: InquiryLabel
    confidence_level: ConfidenceLevel
    rationale: str
    is_compound: bool = False
    sub_labels: List[str] = field(default_factory=list)


@dataclass
class AnswerEvaluation:
    answer: str
    answer_confidence: str   # "high" | "medium" | "low"
    uncertain_parts: str


@dataclass
class AgentResponse:
    strategy: Strategy
    should_respond: bool
    label: InquiryLabel
    confidence_level: ConfidenceLevel
    answer: Optional[str]
    reasoning: str
    is_compound: bool = False
    sub_labels: List[str] = field(default_factory=list)
    answer_confidence: str = ""
    uncertain_parts: str = ""
    tool_result: Optional[Dict] = None   # TOOL_ACTION 실행 결과
    tool_type: str = ""                  # "auto" | "approval" | ""


# ──────────────────────────────────────────────────────────────────
# VectorStore — FAISS + OpenAI Embedding 기반 유사도 검색
# ──────────────────────────────────────────────────────────────────

class VectorStore:
    EMBED_MODEL = "text-embedding-3-large"
    EMBED_DIM   = 3072  # text-embedding-3-large 기본 차원

    def __init__(self, openai_client: AzureOpenAI, cache_path: str = None):
        self._client    = openai_client
        self._cache_path = cache_path
        self._emb_cache: Dict[str, List[float]] = self._load_cache()
        self.payloads: List[Dict] = []   # 각 벡터에 대응하는 메타데이터
        self._texts: List[str]   = []   # 임베딩할 원본 텍스트
        self.index: Optional[faiss.Index] = None

    # ── 캐시 I/O ─────────────────────────────────────────────────

    def _load_cache(self) -> Dict:
        if self._cache_path and os.path.exists(self._cache_path):
            with open(self._cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        if self._cache_path:
            with open(self._cache_path, 'wb') as f:
                pickle.dump(self._emb_cache, f)

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    # ── 문서 추가 / 인덱스 빌드 ──────────────────────────────────

    def add_document(self, text: str, payload: Dict):
        """문서 하나 등록. build_index() 호출 전까지 실제 임베딩은 안 함."""
        self._texts.append(text[:8000])
        self.payloads.append(payload)

    def build_index(self):
        """미캐시 텍스트를 배치 임베딩 후 FAISS IndexFlatIP 구축."""
        import time as _time
        global faiss
        if faiss is None:
            t0 = _time.time()
            print("  faiss 로딩 중...", end="", flush=True)
            import faiss as _faiss
            faiss = _faiss
            print(f" 완료 ({_time.time()-t0:.1f}s)")
        if not self._texts:
            return

        # 캐시에 없는 것만 배치 API 호출
        to_embed = [(self._key(t), t) for t in self._texts
                    if self._key(t) not in self._emb_cache]

        if to_embed:
            total = len(to_embed)
            print(f"  임베딩 API 호출: {total}건 (캐시 미적중)")
            chunk = 100  # OpenAI 배치 최대
            done = 0
            for i in range(0, total, chunk):
                batch = to_embed[i:i + chunk]
                resp = self._client.embeddings.create(
                    model=self.EMBED_MODEL,
                    input=[t for _, t in batch],
                )
                for (key, _), emb in zip(batch, resp.data):
                    self._emb_cache[key] = emb.embedding
                done += len(batch)
                print(f"    임베딩 진행: {done}/{total}", flush=True)
            self._save_cache()
        else:
            print(f"  임베딩 캐시 100% 히트 ({len(self._texts)}건)")

        # FAISS 인덱스 구축
        matrix = np.array(
            [self._emb_cache[self._key(t)] for t in self._texts],
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms == 0, 1, norms)   # L2 정규화 → inner product = cosine

        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        print(f"  FAISS 인덱스 구축 완료: {self.index.ntotal}개 벡터")

    # ── 검색 ─────────────────────────────────────────────────────

    def search(self, query: str, label: str = None, top_k: int = 3,
               similarity_only: bool = False) -> List[Dict]:
        """
        query를 임베딩 후 유사 문서 검색.

        similarity_only=False (기본):
            label 지정 시 동일 라벨 문서를 우선 반환, 부족하면 label=None 문서로 보충.
        similarity_only=True:
            label 무시, 순수 코사인 유사도 상위 top_k 반환.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # 쿼리 임베딩 (캐시 활용)
        key = self._key(query[:8000])
        if key not in self._emb_cache:
            resp = self._client.embeddings.create(model=self.EMBED_MODEL, input=query[:8000])
            self._emb_cache[key] = resp.data[0].embedding
            self._save_cache()

        q = np.array(self._emb_cache[key], dtype=np.float32)
        q /= max(np.linalg.norm(q), 1e-8)
        q = q.reshape(1, -1)

        k = min(top_k if similarity_only else top_k * 6, self.index.ntotal)
        scores, idxs = self.index.search(q, k)

        if similarity_only:
            # 순수 유사도 순 — label 무관하게 top_k
            result = []
            for score, i in zip(scores[0], idxs[0]):
                if i >= 0:
                    result.append({**self.payloads[i], "score": float(score)})
            return result[:top_k]

        # label-aware: 동일 라벨 우선, label=None 보조
        matched, others = [], []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            p = {**self.payloads[i], "score": float(score)}
            doc_label = self.payloads[i].get("label")
            if label and doc_label == label:
                matched.append(p)
            elif not doc_label:
                others.append(p)

        result = matched[:top_k]
        if len(result) < top_k:
            result += others[:top_k - len(result)]
        return result[:top_k]


# ──────────────────────────────────────────────────────────────────
# InquiryAgent
# ──────────────────────────────────────────────────────────────────

class InquiryAgent:
    ADMIN_IDS = {2, 7, 61, 442, 2425, 3417}

    def __init__(self, knowledge_base_path: str = None):
        self.kb       = self._load_knowledge_base(knowledge_base_path)
        self.schedule = self._load_schedule()
        self.user_db  = UserContextDB()
        self.inquiry_history: List[Dict] = []
        # Chat 클라이언트 — Azure OpenAI 02 (gpt-5.2)
        self._client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-12-01-preview",
        )
        # Embedding 클라이언트 — Azure OpenAI 01 (text-embedding-3-large)
        self._embed_client = AzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_EMBED_API_KEY"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT"),
            api_version="2024-12-01-preview",
        )
        self.vector_store: Optional[VectorStore] = None

    # ── Knowledge Base 로드 ────────────────────────────────────────

    def _load_knowledge_base(self, path: str) -> Dict:
        """knowledge_base.json 로드. 없으면 빈 구조 반환."""
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_base.json')
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"prior_knowledge": {}, "label_examples": {}, "error_solutions": []}

    def _load_schedule(self) -> Dict:
        """schedule.json 로드. 없으면 빈 딕셔너리 반환."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'schedule.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    # ── 유틸리티 ──────────────────────────────────────────────────

    def _strip_html(self, html: str) -> str:
        return html_to_text(html)

    def detect_language(self, text: str) -> str:
        if re.search(r'[가-힣]', text):
            return 'ko'
        elif re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
            return 'jp'
        return 'en'

    # ── 사전 지식 요약 (분류·답변 프롬프트 공통 주입) ────────────

    def _prior_knowledge_section(self) -> str:
        """knowledge_base.json의 prior_knowledge를 프롬프트용 텍스트로 변환."""
        pk = self.kb.get("prior_knowledge", {})
        if not pk:
            return ""

        lines = [
            f"## 플랫폼 사전 지식 ({pk.get('platform', 'AI Talent Lab')})",
            pk.get("description", ""),
            "",
            "### 교육 과정",
        ]
        for prog in pk.get("programs", []):
            lines.append(f"- {prog['name']} ({prog['level']}): {prog['description']}")
            if "modules" in prog:
                lines.append(f"  모듈: {', '.join(prog['modules'])}")
            fa = prog.get("final_assignment", {})
            if fa:
                lines.append(f"  최종과제: 강의 {fa.get('required_lectures')}개 수강 완료 후 시작 가능. "
                              f"재제출={'가능' if fa.get('submittable_multiple_times') else '불가'}. "
                              f"{fa.get('note', '')}")
        lines += ["", "### 주요 사실"]
        for fact in pk.get("important_facts", []):
            lines.append(f"- {fact}")

        return "\n".join(lines)

    def _schedule_section(self) -> str:
        """schedule.json의 일정 정보를 프롬프트용 텍스트로 변환."""
        if not self.schedule:
            return ""

        lines = ["## 현재 운영 일정 (동적 정보 — 기수마다 갱신)"]

        lt = self.schedule.get("literacy", {})
        if lt:
            cp = lt.get("course_period", {})
            lines += [
                f"### AI Literacy",
                f"- 수강 기간: {cp.get('start', '미정')} ~ {cp.get('end', '미정')}  ({cp.get('note', '')})",
            ]
            ce = lt.get("certification_exam", {})
            if ce:
                ep = ce.get("enrollment_period", {})
                lines += [
                    f"- 인증시험 {ce.get('current_round', '')}: 시험일 {ce.get('exam_date', '미정')}  |  신청 기간: {ep.get('start', '미정')} ~ {ep.get('end', '미정')}  |  {ce.get('status', '')}",
                    f"  ({ce.get('note', '')})",
                ]

        bc = self.schedule.get("bootcamp", {})
        if bc and bc.get("current_cohort"):
            cohort = bc["current_cohort"]
            ep = bc.get("enrollment_period", {})
            cp = bc.get("course_period", {})
            cycle = bc.get("cohort_cycle_weeks")
            cycle_str = f"  (기수 주기: {cycle}주)" if cycle else ""
            ep_start = ep.get('start') or '미정'
            ep_end   = ep.get('end')   or '미정'
            ep_line  = (f"- 모집 기간: {ep_start} ~ {ep_end}  ({ep.get('note', '')})"
                        if ep_start != '미정' or ep_end != '미정'
                        else "- 모집 기간: 미정")
            lines += [
                f"### AI Bootcamp {cohort}{cycle_str}",
                ep_line,
                f"- 수강 기간: {cp.get('start', '미정')} ~ {cp.get('end', '미정')}  ({cp.get('note', '')})",
                f"- 과제 제출 마감: {bc.get('assignment_deadline', '미정')}",
                f"- 결과 발표 예정: {bc.get('result_announcement', '미정')}",
            ]
            if bc.get("notice"):
                lines.append(f"- 공지: {bc['notice']}")
            for uc in bc.get("upcoming_cohorts", []):
                uc_ep = uc.get("enrollment_period", {})
                uc_cp = uc.get("course_period", {})
                lines += [
                    f"#### 예정 {uc['cohort']} ({uc.get('status', '')})",
                    f"  - 신청 기간: {uc_ep.get('start', '미정')} ~ {uc_ep.get('end', '미정')}",
                    f"  - 수강 기간: {uc_cp.get('start', '미정')} ~ {uc_cp.get('end', '미정')}",
                ]

        mp = self.schedule.get("master_project", {})
        if mp and mp.get("current_cohort"):
            cohort = mp["current_cohort"]
            rp = mp.get("recruitment_period", {})
            cp = mp.get("course_period", {})
            lines += [
                f"### AI Master Project {cohort}",
                f"- 모집 기간: {rp.get('start', '미정')} ~ {rp.get('end', '미정')}",
                f"- 수강 기간: {cp.get('start', '미정')} ~ {cp.get('end', '미정')}",
            ]
            fs = mp.get("final_schedule", {})
            if fs:
                if fs.get("submission_deadline"):
                    lines.append(f"- 최종 산출물 제출 마감: {fs['submission_deadline']}")
                if fs.get("ai_interview_period"):
                    lines.append(f"- AI 면접: {fs['ai_interview_period']}")
                if fs.get("mentor_evaluation"):
                    lines.append(f"- 멘토단 평가: {fs['mentor_evaluation']}")
                if fs.get("result_announcement"):
                    lines.append(f"- 최종 결과 발표: {fs['result_announcement']}")
            if mp.get("notice"):
                lines.append(f"- 공지: {mp['notice']}")
        elif mp and mp.get("notice"):
            lines.append(f"### AI Master Project: {mp['notice']}")

        if self.schedule.get("global_notice"):
            lines += ["", f"### 전체 공지", self.schedule["global_notice"]]

        return "\n".join(lines)

    # ── 분류 프롬프트용 라벨 설명 ─────────────────────────────────

    def _label_description_section(self) -> str:
        """knowledge_base.json의 label_examples를 분류 프롬프트용 텍스트로 변환."""
        label_examples = self.kb.get("label_examples", {})

        sections = []
        all_labels = [
            ("ACCOUNT_ACTION_REQUIRED", "운영자 직접 조치"),
            ("PLATFORM_SYSTEM_ERROR",   "플랫폼 서버·시스템 에러"),
            ("VIDEO_PLAYBACK_ERROR",    "강의 영상 재생 안됨"),
            ("FEATURE_REQUEST",         "기능 개선·건의"),
            ("UNCATEGORIZED",           "불명확·분류 불가"),
            ("COURSE_INFO",             "강의·커리큘럼 정보"),
            ("SUBMISSION_POLICY",       "과제 제출 정책"),
            ("SERVICE_GUIDE",           "플랫폼 이용 가이드"),
            ("ASSIGNMENT_DEVELOPMENT",  "과제 개발 방향"),
            ("CODE_LOGIC_ERROR",        "코드·API 오류"),
            ("CODE_REVIEW_RESET",       "코드 리뷰 횟수 초기화"),
            ("LITERACY_PRACTICE_RESET", "AI Literacy 사전연습 횟수 복구"),
        ]

        for label_key, short_name in all_labels:
            info = label_examples.get(label_key, {})
            desc = info.get("description", "")
            patterns = info.get("typical_patterns", [])
            examples = info.get("qa_examples", [])[:2]  # 최대 2개 예시

            block = [f"- {label_key} ({short_name}): {desc}"]
            if patterns:
                block.append(f"  패턴: {' / '.join(patterns[:3])}")
            if examples:
                ex = examples[0]
                block.append(f"  예시: 제목=\"{ex['title']}\"")
            sections.append("\n".join(block))

        return "\n".join(sections)

    # ── Step 1: LLM 분류 ──────────────────────────────────────────

    def _llm_classify(self, inquiry: Dict,
                      personal_context: str = "") -> LLMClassification:
        title   = inquiry.get('title', '')
        content = self._strip_html(inquiry.get('content', ''))

        prior_knowledge = self._prior_knowledge_section()
        schedule        = self._schedule_section()
        label_descs     = self._label_description_section()

        personal_section = f"\n{personal_context}\n" if personal_context else ""

        # label_descs를 그룹별로 분리
        _g1_end = label_descs.find('- COURSE_INFO')
        _g2_end = label_descs.find('- CODE_REVIEW_RESET')
        label_descs_g1 = label_descs[:_g1_end].strip() if _g1_end != -1 else label_descs
        label_descs_g2 = label_descs[_g1_end:_g2_end].strip() if _g1_end != -1 and _g2_end != -1 else ""
        label_descs_g3 = label_descs[_g2_end:].strip() if _g2_end != -1 else ""

        system_prompt = f"""너는 AI Talent Lab 문의 분류 Agent야.{personal_section}
문의를 읽고 아래 카테고리 중 하나로 분류하고, 신뢰도를 판단해.

{prior_knowledge}

{schedule}

## 카테고리 (label) 정의

### Group 1 — 운영자 에스컬레이션 (RAG 답변 생성 안 함):
{label_descs_g1}

### Group 2 — RAG 기반 답변 시도:
{label_descs_g2}

### Group 3 — 에이전트 직접 실행 (DB 조작):
{label_descs_g3}

## 분류 판단 기준

**ACCOUNT_ACTION_REQUIRED vs PLATFORM_SYSTEM_ERROR 구분:**
- 특정 사용자의 계정·권한·버튼 활성화를 운영자가 직접 바꿔줘야 → ACCOUNT_ACTION_REQUIRED
- 플랫폼 시스템 자체의 버그·장애 (콘솔 접근 불가, 무한 로딩, 배포 오류) → PLATFORM_SYSTEM_ERROR

**SUBMISSION_POLICY vs COURSE_INFO 구분:**
- 과제 제출 횟수·마감·재제출·평가 결과 발표 시기 → SUBMISSION_POLICY
- 강의 이수 조건·커리큘럼·수료 관계 → COURSE_INFO

**CODE_LOGIC_ERROR vs ASSIGNMENT_DEVELOPMENT 구분:**
- 코드 에러 메시지, API 호출 실패, 라이브러리 오류 → CODE_LOGIC_ERROR
- 과제 설계 방향, 아키텍처, 구현 접근법 → ASSIGNMENT_DEVELOPMENT

**CODE_REVIEW_RESET 판단:**
- "코드 리뷰 횟수 초기화", "리뷰 횟수 리셋", "리뷰 다시", "코드 리뷰 초기화" 등 명시적 요청 → CODE_REVIEW_RESET
- ACCOUNT_ACTION_REQUIRED와 혼동 주의: 코드 리뷰 횟수 리셋은 에이전트가 직접 처리 가능하므로 CODE_REVIEW_RESET으로 분류

**LITERACY_PRACTICE_RESET 판단:**
- "AI Literacy 사전연습 횟수 복구", "모의 인증 횟수 원복", "실습 횟수 잘못 차감", "연습 횟수 돌려주세요" 등 → LITERACY_PRACTICE_RESET
- ACCOUNT_ACTION_REQUIRED와 혼동 주의: 사전연습 횟수 복구는 에이전트가 직접 처리 가능하므로 LITERACY_PRACTICE_RESET으로 분류

## 신뢰도 (confidence_level) 판단 요소:
① 문의 명확성   — 무엇을 묻는지 텍스트만 봐도 알 수 있는가
② 카테고리 단일성 — 10개 중 딱 하나에만 해당하는가
   (복합 문의로 sub_label을 명확히 식별 가능한 경우는 ② 충족으로 간주)

레벨:
- high : ①② 모두 충족
- low  : ① 또는 ② 미충족 — 아래 중 하나에 해당:
         · 문의 내용이 너무 짧거나 불명확해 무엇을 묻는지 파악 불가
         · 두 카테고리 모두 동등하게 해당되어 분류 불가
         · 운영자만 접근 가능한 개인 데이터(점수·처리 상태·계정 이력)가 있어야만 답변 가능

## 복합 문의 감지
문의 내에 성격이 서로 다른 카테고리의 질문이 2개 이상 포함된 경우:
- is_compound = true 로 설정
- sub_labels 에 감지된 모든 라벨을 나열
- label 은 sub_labels 중 가장 우선순위가 높은 라벨로 설정 (Group1 라벨 우선)
예: Q1=COURSE_INFO, Q2=ACCOUNT_ACTION_REQUIRED → label="ACCOUNT_ACTION_REQUIRED", is_compound=true, sub_labels=["COURSE_INFO","ACCOUNT_ACTION_REQUIRED"]

## 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{
  "label": "LABEL_NAME",
  "confidence_level": "high" | "low",
  "rationale": "분류 근거 한 줄",
  "is_compound": false,
  "sub_labels": []
}}"""

        user_content = f"제목: {title}\n내용: {content}"

        try:
            response = self._client.chat.completions.create(
                model=os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-5.2"),
                max_completion_tokens=300,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(json_match.group() if json_match else raw)

            label      = InquiryLabel(result.get("label", "UNCATEGORIZED"))
            confidence = ConfidenceLevel(result.get("confidence_level", "low"))
            is_compound = bool(result.get("is_compound", False))
            sub_labels  = result.get("sub_labels", [])
            return LLMClassification(
                label=label,
                confidence_level=confidence,
                rationale=result.get("rationale", ""),
                is_compound=is_compound,
                sub_labels=sub_labels,
            )

        except Exception as e:
            return LLMClassification(
                label=InquiryLabel.UNCATEGORIZED,
                confidence_level=ConfidenceLevel.LOW,
                rationale=f"분류 중 오류: {e}",
            )

    # ── Step 2: Strategy 결정 ─────────────────────────────────────

    def _determine_strategy(
        self, label: InquiryLabel, confidence: ConfidenceLevel,
        classification: 'LLMClassification' = None,
    ) -> Tuple[Strategy, bool]:
        # 복합 문의 처리
        if classification and classification.is_compound and len(classification.sub_labels) >= 2:
            valid_sub = list(dict.fromkeys(          # 순서 보존 dedup
                InquiryLabel(sl) for sl in classification.sub_labels
                if sl in {l.value for l in InquiryLabel}
            ))
            if len(valid_sub) >= 2:
                # Group1 포함 → human_review (운영자 직접 조치 필요)
                if any(sl in GROUP1 for sl in valid_sub):
                    return Strategy.HUMAN_REVIEW, True
                # Group2만 4개 이상 → human_review (복잡도 초과)
                if len(valid_sub) > 3:
                    return Strategy.HUMAN_REVIEW, True
                # Group2만 2-3개 → tool_rag (각 label별 RAG 합산, process_inquiry에서 처리)
                return Strategy.TOOL_RAG, True

        if label.value in GROUP3_LABELS:
            return Strategy.TOOL_ACTION, True
        if label in GROUP1 or confidence == ConfidenceLevel.LOW:
            return Strategy.NO_RESPONSE, False
        return Strategy.TOOL_RAG, True

    # ── RAG: FAISS 벡터 검색 기반 KB 검색 ───────────────────────

    def _build_kb_context(self, label: InquiryLabel, inquiry_text: str,
                          similarity_only: bool = False) -> Tuple[str, float]:
        """
        FAISS 벡터 검색으로 해당 label의 유사 Q&A 최대 5개 + 에러 솔루션 보완.
        vector_store가 없으면 키워드 폴백 사용.
        similarity_only=True 이면 label 무시하고 순수 유사도 상위 문서를 가져옴.
        반환: (context 문자열, 최고 유사도 점수)  — 유사도가 없으면 0.0
        """
        parts: List[str] = []
        text_lower = inquiry_text.lower()
        max_score = 0.0

        # ① FAISS 벡터 검색
        if self.vector_store and self.vector_store.index:
            hits = self.vector_store.search(inquiry_text, label=label.value, top_k=5,
                                            similarity_only=similarity_only)
            for hit in hits:
                max_score = max(max_score, hit.get("score", 0.0))
                q_short = hit.get("title", "")
                a_short = hit.get("answer", "")[:300]
                tag = f"[유사 예제 ({hit.get('type','?')} | score={hit['score']:.3f})]"
                parts.append(f"{tag}\nQ: {q_short}\nA: {a_short}")
        else:
            # 폴백: 키워드 토큰 overlap
            label_info = self.kb.get("label_examples", {}).get(label.value, {})
            inquiry_tokens = set(re.findall(r'[가-힣a-zA-Z0-9]+', text_lower))
            examples = sorted(
                label_info.get("qa_examples", []),
                key=lambda ex: len(inquiry_tokens & set(
                    re.findall(r'[가-힣a-zA-Z0-9]+',
                               (ex.get("title","") + ex.get("question","")).lower()))),
                reverse=True,
            )
            for ex in examples[:2]:
                parts.append(f"[라벨 예제: {label.value}]\nQ: {ex['title']}\n"
                             f"{ex['question'][:200]}\nA: {ex['answer'][:300]}")

        # ② 에러 솔루션 정규식 보완 (CODE_LOGIC_ERROR 또는 에러 키워드 감지 시)
        if label == InquiryLabel.CODE_LOGIC_ERROR or re.search(r'error|traceback|오류|에러|리뷰.*실패|실패.*리뷰|venv', text_lower):
            for sol in self.kb.get("error_solutions", []):
                if re.search(sol["error_pattern"], inquiry_text, re.I):
                    parts.append(f"[에러 가이드]\n{sol['title']}: {sol['solution']}")
                    break

        # ③ 과정 정보 보완 (COURSE_INFO 라벨)
        if label == InquiryLabel.COURSE_INFO:
            for prog in self.kb.get("prior_knowledge", {}).get("programs", []):
                if any(k in text_lower for k in prog.get("keywords", [])):
                    fa = prog.get("final_assignment", {})
                    desc = f"{prog['name']} ({prog['level']}): {prog['description']}"
                    if fa:
                        desc += (f"\n  최종과제: 강의 {fa.get('required_lectures')}개 완료 후 시작. "
                                 f"{fa.get('note', '')}")
                    parts.append(f"[과정 정보]\n{desc}")

        return "\n\n".join(parts) if parts else "관련 KB 정보 없음", max_score

    # ── 답변 생성 ─────────────────────────────────────────────────

    def _generate_answer(
        self,
        inquiry: Dict,
        label: InquiryLabel,
        kb_context: str,
        compound_labels: List[InquiryLabel] = None,
        personal_context: str = "",
    ) -> AnswerEvaluation:
        title   = inquiry.get('title', '')
        content = self._strip_html(inquiry.get('content', ''))
        lang    = self.detect_language(title + " " + content)

        prior_knowledge = self._prior_knowledge_section()
        schedule        = self._schedule_section()

        greetings = {
            'ko': '안녕하세요, AI Talent Lab입니다.',
            'en': 'Hello, this is AI Talent Lab.',
            'jp': 'こんにちは、AI Talent Labです。',
        }

        if compound_labels and len(compound_labels) >= 2:
            labels_str = " / ".join(l.value for l in compound_labels)
            category_line = f"이 문의는 복합 문의({labels_str})야. KB 정보의 각 섹션을 참고해서 질문 순서대로 빠짐없이 답변해줘."
        else:
            category_line = f"이 문의는 \"{label.value}\" 카테고리로 분류되었어."

        personal_section = f"\n{personal_context}\n" if personal_context else ""

        system_prompt = f"""너는 AI Talent Lab 문의 답변 Agent야.
{personal_section}
{prior_knowledge}

{schedule}

[답변 규칙]
- {category_line}
- 언어: {'한국어' if lang == 'ko' else '영어' if lang == 'en' else '일본어'}
- 인사말로 시작: {greetings.get(lang, greetings['ko'])}
- 아래 참고 정보를 우선 활용해서 답변해.
- 확인되지 않은 내용은 절대 추측하지 말고 "확인 후 안내드리겠습니다"로 마무리해.
- 답변에 "KB", "참고 정보", "시스템" 같은 내부 용어를 절대 노출하지 마. 수강생이 보는 답변이야.
- 답변은 간결하고 명확하게. 마지막에 "감사합니다." 로 끝낼 것.

[답변 신뢰도 자체 평가]
답변 작성 후, 아래 기준으로 answer_confidence를 판단해:
- high  : 참고 정보에 명확한 근거가 있어 답변이 완결됨
- medium: 참고 정보로 부분 답변 가능하나 운영자 확인이 필요한 부분 존재
- low   : 참고 정보에 근거가 부족해 추측성 답변이 됨

## 출력 형식 (JSON만 출력, 다른 텍스트 없이)
{{
  "answer": "답변 텍스트",
  "answer_confidence": "high" | "medium" | "low",
  "uncertain_parts": "확신 없는 부분 설명 (없으면 빈 문자열)"
}}"""

        user_content = f"""[문의]
제목: {title}
내용: {content}

[참고 정보]
{kb_context}"""

        try:
            response = self._client.chat.completions.create(
                model=os.environ.get("AZURE_CHAT_DEPLOYMENT", "gpt-5.2"),
                max_completion_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            result = json.loads(json_match.group() if json_match else raw)
            return AnswerEvaluation(
                answer=result.get("answer", ""),
                answer_confidence=result.get("answer_confidence", "medium"),
                uncertain_parts=result.get("uncertain_parts", ""),
            )
        except Exception as e:
            return AnswerEvaluation(
                answer=f"답변 생성 오류: {e}",
                answer_confidence="low",
                uncertain_parts=str(e),
            )

    # ── 메인 처리 흐름 ────────────────────────────────────────────

    def process_inquiry(self, inquiry: Dict) -> AgentResponse:
        title   = inquiry.get('title', '')
        content = self._strip_html(inquiry.get('content', ''))
        inquiry_text = title + " " + content

        # 개인 맥락 조회 (수강 이력)
        personal_context = self.user_db.build_personal_context_str(
            inquiry.get('author_id')
        )

        # Step 1: LLM 분류
        classification = self._llm_classify(inquiry, personal_context=personal_context)

        # Step 2: Strategy 결정
        strategy, should_respond = self._determine_strategy(
            classification.label, classification.confidence_level, classification
        )

        if not should_respond:
            return AgentResponse(
                strategy=strategy,
                should_respond=False,
                label=classification.label,
                confidence_level=classification.confidence_level,
                answer=None,
                reasoning=classification.rationale,
                is_compound=classification.is_compound,
                sub_labels=classification.sub_labels,
            )

        # Step 3: Tool Action (GROUP3) — tools.py 실행기 위임
        if strategy == Strategy.TOOL_ACTION:
            answer_text, tool_result, tool_type = execute_tool_action(
                classification.label.value,
                inquiry.get("author_id"),
                inquiry_text,
                self.user_db,
            )
            self.inquiry_history.append({
                "inquiry":          inquiry,
                "label":            classification.label.value,
                "strategy":         strategy.value,
                "has_admin_answer": False,
                "admin_answers":    [],
            })
            return AgentResponse(
                strategy=strategy,
                should_respond=True,
                label=classification.label,
                confidence_level=classification.confidence_level,
                answer=answer_text,
                reasoning=classification.rationale,
                answer_confidence="high" if tool_result.get("success") else "low",
                tool_result=tool_result,
                tool_type=tool_type,
            )

        # RAG 검색
        compound_labels_for_rag = None

        # 복합 문의 (Group2만, 2-3개): sub_labels 각각 RAG 후 context 합산
        if classification.is_compound and strategy == Strategy.TOOL_RAG:
            valid_sub = list(dict.fromkeys(
                InquiryLabel(sl) for sl in classification.sub_labels
                if sl in {l.value for l in InquiryLabel}
            ))
            if len(valid_sub) >= 2 and all(sl in GROUP2 for sl in valid_sub):
                ctx_parts, scores = [], []
                for sub_lbl in valid_sub:
                    ctx, score = self._build_kb_context(sub_lbl, inquiry_text, similarity_only=True)
                    ctx_parts.append(f"[{sub_lbl.value} 관련]\n{ctx}")
                    scores.append(score)
                kb_context = "\n\n---\n\n".join(ctx_parts)
                max_rag_score = min(scores)   # 가장 낮은 score 기준으로 다운그레이드 판단
                compound_labels_for_rag = valid_sub
            else:
                kb_context, max_rag_score = self._build_kb_context(classification.label, inquiry_text, similarity_only=True)
        else:
            kb_context, max_rag_score = self._build_kb_context(classification.label, inquiry_text, similarity_only=True)

        # 1차 strategy 결정: RAG 유사도 기반 (tool_rag인 경우에만)
        if strategy == Strategy.TOOL_RAG and max_rag_score < RAG_CONFIDENCE_THRESHOLD:
            strategy = Strategy.HUMAN_REVIEW
            classification.rationale += (
                f" [RAG 유사도 낮음: {max_rag_score:.3f} < {RAG_CONFIDENCE_THRESHOLD}]"
            )

        # 답변 생성 + 자체 평가 (LLM이 실제 KB context를 보고 신뢰도 판단)
        eval_result = self._generate_answer(
            inquiry, classification.label, kb_context,
            compound_labels=compound_labels_for_rag,
            personal_context=personal_context,
        )

        # 2차 분기: 답변 신뢰도 기반 (다운그레이드만, 업그레이드 없음)
        answer_conf = eval_result.answer_confidence
        if answer_conf == "low":
            strategy = Strategy.NO_RESPONSE
        elif answer_conf == "medium" and strategy == Strategy.TOOL_RAG:
            strategy = Strategy.HUMAN_REVIEW

        # human_review면 [초안] 태그 추가
        answer_text = eval_result.answer
        if strategy == Strategy.HUMAN_REVIEW:
            answer_text = f"[초안] {answer_text}"

        # human_review 이후 should_respond 결정
        if strategy == Strategy.NO_RESPONSE:
            self.inquiry_history.append({
                'inquiry': inquiry,
                'label': classification.label.value,
                'strategy': strategy.value,
                'has_admin_answer': False,
                'admin_answers': [],
            })
            return AgentResponse(
                strategy=strategy,
                should_respond=False,
                label=classification.label,
                confidence_level=classification.confidence_level,
                answer=None,
                reasoning=classification.rationale + f" [답변 신뢰도 낮음: {eval_result.uncertain_parts}]",
                is_compound=classification.is_compound,
                sub_labels=classification.sub_labels,
                answer_confidence=answer_conf,
                uncertain_parts=eval_result.uncertain_parts,
            )

        # history 기록
        self.inquiry_history.append({
            'inquiry': inquiry,
            'label': classification.label.value,
            'strategy': strategy.value,
            'has_admin_answer': False,
            'admin_answers': [],
        })

        return AgentResponse(
            strategy=strategy,
            should_respond=True,
            label=classification.label,
            confidence_level=classification.confidence_level,
            answer=answer_text,
            reasoning=classification.rationale,
            is_compound=classification.is_compound,
            sub_labels=classification.sub_labels,
            answer_confidence=answer_conf,
            uncertain_parts=eval_result.uncertain_parts,
        )

    # ── history 로드 (train + test 모두 지원) ─────────────────────

    def load_inquiry_history(
        self,
        inquiry_data: List[Dict],
        comment_data: List[Dict],
        pre_label: bool = False,
    ):
        """
        과거 문의 이력 로드 (RAG용).
        pre_label=True 이면 label_examples 키워드 패턴으로 라벨 사전 부여 (휴리스틱).
        로드 완료 후 자동으로 FAISS 벡터 인덱스를 구축함.
        """
        # comment_data를 inquiry_id별로 미리 묶어서 O(n+m)으로 처리
        comment_map: Dict[int, List[Dict]] = {}
        for c in comment_data:
            if c.get('author_id') in self.ADMIN_IDS:
                comment_map.setdefault(c['inquiry_id'], []).append(c)

        for inquiry in inquiry_data:
            admin_comments = comment_map.get(inquiry['id'], [])
            label = None
            if pre_label:
                label = self._heuristic_label(inquiry)

            self.inquiry_history.append({
                'inquiry':        inquiry,
                'label':          label,
                'has_admin_answer': len(admin_comments) > 0,
                'admin_answers':  admin_comments,
            })

        # history 로드 후 벡터 인덱스 자동 구축
        self.build_vector_index()

    def build_vector_index(self):
        """
        KB 큐레이션 예제 + 에러 솔루션 + history(운영자 답변 있는 것)를
        FAISS IndexFlatIP로 색인.
        embeddings_cache.pkl 에 임베딩 결과를 캐시하므로 재실행 시 API 미호출.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(base, "embeddings_cache.pkl")
        vs = VectorStore(self._embed_client, cache_path=cache_path)

        # ① KB 큐레이션 Q&A
        for label_key, info in self.kb.get("label_examples", {}).items():
            for ex in info.get("qa_examples", []):
                text = ex.get("title", "") + "\n" + ex.get("question", "")
                vs.add_document(text, {
                    "label":  label_key,
                    "title":  ex.get("title", ""),
                    "answer": ex.get("answer", ""),
                    "type":   "kb_curated",
                })

        # ② 에러 솔루션
        for sol in self.kb.get("error_solutions", []):
            text = sol.get("title", "") + "\n" + sol.get("solution", "")
            vs.add_document(text, {
                "label":  "CODE_LOGIC_ERROR",
                "title":  sol.get("title", ""),
                "answer": sol.get("solution", ""),
                "type":   "error_solution",
            })

        # ③ History (운영자 답변 있는 것만)
        for h in self.inquiry_history:
            if not h.get("has_admin_answer") or not h.get("admin_answers"):
                continue
            inq   = h.get("inquiry", {})
            title = inq.get("title", "")
            body  = self._strip_html(inq.get("content", ""))[:500]
            ans   = self._strip_html(h["admin_answers"][0].get("content", ""))[:400]
            vs.add_document(title + "\n" + body, {
                "label":  h.get("label"),   # 휴리스틱 라벨 or None
                "title":  title,
                "answer": ans,
                "type":   "history",
            })

        print(f"벡터 인덱스 구축 중... (총 {len(vs._texts)}개 문서)")
        vs.build_index()
        self.vector_store = vs

    def _heuristic_label(self, inquiry: Dict) -> Optional[str]:
        """
        키워드 기반 라벨 휴리스틱 (LLM 호출 없이 사전 분류).
        정확도보다 속도 우선 — RAG history 검색에만 사용.
        """
        text = (inquiry.get('title', '') + ' ' +
                html_to_text(inquiry.get('content', ''))).lower()

        rules = [
            # (패턴, 라벨) — 순서가 중요: 더 구체적인 패턴 먼저
            # ① 계정·권한 직접 조치
            (r'인증.*버튼|버튼.*비활성|비활성화|수강.*버튼|접근.*안\s*됩|수강.*기간|권한.*변경|응시.*자격|입장.*불가|수강.*불가',
             "ACCOUNT_ACTION_REQUIRED"),
            # ② 플랫폼 시스템 오류
            (r'콘솔.*접근|python.*실행.*안|스크립트.*실행.*안|무한.*로딩|로딩.*무한|서버.*장애|ide.*실행|streamlit.*실행.*안|배포.*오류|파일.*실행.*안',
             "PLATFORM_SYSTEM_ERROR"),
            # ③ 영상 재생
            (r'영상.*재생|동영상.*로딩|강의.*영상.*안|동영상.*안\s*나|영상.*안\s*보',
             "VIDEO_PLAYBACK_ERROR"),
            # ④ 기능 건의
            (r'기능.*추가|개선.*요청|건의|ui.*변경|문구.*변경|다운로드.*요청|파일.*다운|공유.*요청',
             "FEATURE_REQUEST"),
            # ⑤ 코드·API 오류 (에러 메시지가 텍스트에 있음)
            (r'traceback|error:|keyerror|typeerror|modulenotfounderror|importerror|attributeerror'
             r'|api.?key|credentials|rate.?limit|오류.*발생|에러.*발생|exception|invoke.*안|파싱.*오류'
             r'|pip install|패키지.*설치|langchain.*오류|openai.*오류',
             "CODE_LOGIC_ERROR"),
            # ⑥ 과제 제출 정책
            (r'재제출|제출.*가능|마감|과제.*결과|수료.*결과|미이수|미수료|평가.*발표|결과.*발표|수료.*여부'
             r'|이수.*처리|제출.*완료.*확인|제출.*기간|과제.*기간',
             "SUBMISSION_POLICY"),
            # ⑦ 강의·커리큘럼 정보
            (r'커리큘럼|모듈.*구성|강의.*몇|강의.*완료.*과제|수강.*신청|ai bootcamp.*수료.*ai literacy'
             r'|수료.*처리|강의.*이수|수강.*방법|과정.*안내|bootcamp.*과정',
             "COURSE_INFO"),
            # ⑧ 플랫폼 사용 가이드
            (r'ide.*사용.*방법|ide.*가이드|콘솔.*사용|실행.*방법.*안내|과제.*제출.*절차|양식.*요청|사용.*방법.*문의',
             "SERVICE_GUIDE"),
            # ⑨ 과제 개발 방향·아키텍처
            (r'sub.?graph|아키텍처|구현.*방법|개발.*방향|rag.*구현|멀티.*에이전트|설계.*방법|폴더.*구조'
             r'|chain.*구성|agent.*구조|과제.*주제|서비스.*기획',
             "ASSIGNMENT_DEVELOPMENT"),
        ]

        for pattern, label in rules:
            if re.search(pattern, text):
                return label

        return None


# ──────────────────────────────────────────────────────────────────
# 유틸리티 / 메인
# ──────────────────────────────────────────────────────────────────

def load_json_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_dotenv():
    """70793/ 또는 상위 디렉토리의 .env 파일에서 환경변수 로드"""
    base = os.path.dirname(os.path.abspath(__file__))
    for dirpath in (base, os.path.dirname(base)):
        env_path = os.path.join(dirpath, '.env')
        if os.path.exists(env_path):
            with open(env_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
            break


def _answer_similarity(vs: 'VectorStore', a: str, b: str) -> float:
    """두 답변 텍스트의 임베딩 코사인 유사도."""
    if not a or not b:
        return 0.0
    for text in (a, b):
        key = vs._key(text[:8000])
        if key not in vs._emb_cache:
            resp = vs._client.embeddings.create(model=vs.EMBED_MODEL, input=text[:8000])
            vs._emb_cache[key] = resp.data[0].embedding
    va = np.array(vs._emb_cache[vs._key(a[:8000])], dtype=np.float32)
    vb = np.array(vs._emb_cache[vs._key(b[:8000])], dtype=np.float32)
    va /= max(np.linalg.norm(va), 1e-8)
    vb /= max(np.linalg.norm(vb), 1e-8)
    return float(np.dot(va, vb))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Talent Lab 문의 Agent PoC")
    parser.add_argument("--n-test",       type=int, default=20,  help="테스트케이스 수 (기본 10)")
    parser.add_argument("--random-state", type=int, default=12,  help="랜덤 시드 (기본 42)")
    args = parser.parse_args()

    import time as _time

    print("=== AI Talent Lab 문의 Agent PoC ===\n")
    print(f"random_state={args.random_state}  n_test={args.n_test}\n", flush=True)

    _load_dotenv()

    if not os.environ.get("AZURE_OPENAI_API_KEY") or not os.environ.get("AZURE_OPENAI_EMBED_API_KEY"):
        print("[오류] AZURE_OPENAI_API_KEY 또는 AZURE_OPENAI_EMBED_API_KEY 환경변수가 없습니다.")
        return

    t0 = _time.time()
    agent = InquiryAgent()
    print(f"[타이머] Agent 초기화: {_time.time()-t0:.1f}s", flush=True)

    base_path = os.path.dirname(os.path.abspath(__file__))

    t0 = _time.time()
    all_inquiries = load_json_file(os.path.join(base_path, 'inquiry_all.json'))
    all_comments  = load_json_file(os.path.join(base_path, 'inquiry_comment_all.json'))
    print(f"[타이머] JSON 로드: {_time.time()-t0:.1f}s", flush=True)
    print(f"통합 문의 데이터: {len(all_inquiries)}건 / 댓글: {len(all_comments)}건", flush=True)

    # 관리자 답변이 있는 문의만 테스트 대상으로 추출 (비교 가능한 것만)
    admin_ids = agent.ADMIN_IDS
    comment_map: Dict[int, List[Dict]] = {}
    for c in all_comments:
        if c.get('author_id') in admin_ids:
            comment_map.setdefault(c['inquiry_id'], []).append(c)

    has_answer = [inq for inq in all_inquiries if inq['id'] in comment_map]

    rng = random.Random(args.random_state)
    n = min(args.n_test, len(has_answer))
    test_cases = rng.sample(has_answer, n)
    test_ids   = {inq['id'] for inq in test_cases}

    # 테스트셋을 제외한 나머지를 RAG pool로 사용 (관리자 답변 있는 것만)
    rag_pool = [inq for inq in all_inquiries if inq['id'] not in test_ids and inq['id'] in comment_map]
    print(f"테스트셋: {len(test_cases)}건 / RAG pool: {len(rag_pool)}건", flush=True)

    t0 = _time.time()
    agent.load_inquiry_history(rag_pool, all_comments, pre_label=False)
    print(f"[타이머] RAG history 로드 + 벡터 인덱스: {_time.time()-t0:.1f}s", flush=True)
    print(f"RAG history 로드: {len(agent.inquiry_history)}건\n")
    print("Agent 준비 완료\n", flush=True)

    strategy_labels = {
        Strategy.NO_RESPONSE:  "운영자 에스컬레이션",
        Strategy.HUMAN_REVIEW: "RAG 초안 + 운영자 검토",
        Strategy.TOOL_RAG:     "RAG 자동 답변 게시",
        Strategy.TOOL_ACTION:  "에이전트 직접 실행",
    }

    for i, test_inquiry in enumerate(test_cases, 1):
        title   = test_inquiry.get('title', '')
        content = agent._strip_html(test_inquiry.get('content', ''))
        content_preview = content[:200].replace('\n', ' ').strip()

        personal_ctx = agent.user_db.build_personal_context_str(test_inquiry.get('author_id'))

        # 실제 관리자 답변
        actual_answers = comment_map.get(test_inquiry['id'], [])
        actual_text    = agent._strip_html(actual_answers[0].get('content', '')) if actual_answers else ''

        print(f"\n{'='*60}")
        print(f"테스트 {i}/{len(test_cases)}: {title}")
        print(f"{'='*60}")
        print(f"[제목]   {title}")
        print(f"[내용]   {content_preview}{'...' if len(content) > 200 else ''}")
        print(f"[날짜]   {test_inquiry.get('create_dt', '없음')}")
        if personal_ctx:
            for line in personal_ctx.splitlines():
                print(f"[DB]     {line}")
        else:
            print(f"[DB]     (수강 이력 없음)")
        print(f"{'-'*60}")

        response = agent.process_inquiry(test_inquiry)

        print(f"[Label]       {response.label.value}")
        print(f"[신뢰도]      {response.confidence_level.value}")
        print(f"[Strategy]    {strategy_labels[response.strategy]}")
        if response.is_compound:
            sub_labels = response.sub_labels
            all_group2 = all(
                sl in {l.value for l in GROUP2} for sl in sub_labels
            )
            multi_rag = all_group2 and 2 <= len(set(sub_labels)) <= 3
            mode = "multi-RAG 합산" if multi_rag else "human_review 라우팅"
            print(f"[복합 문의]   sub_labels={sub_labels}  ({mode})")
        print(f"[판단 근거]   {response.reasoning}")

        # Tool 실행 결과 출력 (Group 3 경로)
        if response.tool_result is not None:
            tr = response.tool_result
            tool_type_label = {"auto": "AUTO_TOOL", "approval": "APPROVAL_TOOL"}.get(response.tool_type, response.tool_type)
            print(f"[Tool 종류]   {tool_type_label}")
            if tr.get("success"):
                print(f"[Tool 결과]   ✅ 성공")
                if "prev_used" in tr:
                    # CODE_REVIEW_RESET
                    print(f"  → 리셋 전 사용 횟수: {tr.get('prev_used', 0)}회")
                    print(f"  → 리셋 후 남은 횟수: {CODE_REVIEW_DAILY_LIMIT}회 (일일 한도)")
                    print(f"  → 대상 lecture_id: {tr.get('lecture_id', 'N/A')}")
                    print(f"  → 누적 리셋 횟수: {tr.get('reset_count', 'N/A')}")
                elif "restored" in tr:
                    # LITERACY_PRACTICE_RESET
                    print(f"  → 복구 횟수: {tr.get('restored', 0)}회")
                    print(f"  → 총 부여: {tr.get('tutorial_attempts', 'N/A')}회 / 사용: {tr.get('tutorial_used', 'N/A')}회")
                    print(f"  → 남은 연습 횟수: {tr.get('remaining', 'N/A')}회")
                    print(f"  → 대상 literacy_test_id: {tr.get('literacy_test_id', 'N/A')}")
                if tr.get("user_identifier"):
                    print(f"  → 사용자: {tr['user_identifier']}")
            else:
                print(f"[Tool 결과]   ❌ 실패")
                print(f"  → 사유: {tr.get('reason', '알 수 없음')}")

        if response.answer:
            print(f"\n[생성된 답변]\n{response.answer}")

        if actual_text:
            print(f"\n[실제 관리자 답변]\n{actual_text[:400]}{'...' if len(actual_text) > 400 else ''}")
            if response.answer and agent.vector_store:
                sim = _answer_similarity(agent.vector_store, response.answer, actual_text)
                print(f"\n[답변 유사도]  {sim*100:.1f}%")


if __name__ == "__main__":
    main()
