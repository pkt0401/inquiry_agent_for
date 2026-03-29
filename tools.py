"""
tools.py — Agent Tool 정의 및 실행기

실서비스(ai-talent-lab/services/inquiry-agent)의 LangChain tool 패턴을 참고:
  - @tool 데코레이터로 LLM이 직접 호출 가능한 tool 정의
  - create_langchain_tools() 팩토리로 user context를 클로저 바인딩
  - 내부 ID 노출 방지, 안전장치 포함

Tool 종류:
  AUTO_TOOL     : 즉시 실행 (저위험, 단순 복구)
  APPROVAL_TOOL : 관리자 승인 후 실행 (고위험·비가역적 액션, 추후 추가)

──────────────────────────────────────────────────────────────────
운영 DB 테이블 매핑 (PoC SQLite → 운영 MySQL)

 Tool                      | 운영 테이블                        | 주요 컬럼                                          | 동작
 ─────────────────────────-┼────────────────────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────
 CODE_REVIEW_RESET         | code_review_log                    | user_id, lecture_id, started_at, status             | COUNT로 당일 사용량 산출 → 리셋 레코드 삽입/갱신
 LITERACY_PRACTICE_RESET   | literacy_test_user_status           | user_id, literacy_test_id, tutorial_attempts,       | tutorial_used 차감 또는 tutorial_attempts 증가
                           |                                    | tutorial_used, status                               |
                           | (또는) user_tutorial_status         | user_id, tutorial_attempts, tutorial_used            | 글로벌 사전연습 횟수 관리

 개인화 컨텍스트 조회 관련:
 ─────────────────────────-┼────────────────────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────
 사용자 정보               | user                               | id, user_id, username, email, group1                | author_id로 사용자 조회
 기수 정보                 | cohort                             | id, type, name, start_date, end_date, program_id    | 프로그램별 기수 목록
 수강 이력                 | user_cohorts                       | user_id, cohort_id, open_dt, close_dt               | 수강 등록 여부
 수강 상태                 | user_cohort_status                 | user_id, cohort_id, status, description             | 수료/미수료/진행중
 프로그램 정보             | program                            | id, title, type                                     | AI Bootcamp / AI Literacy / Master Project
 문의 원본                 | inquiry                            | id, title, content, author_id, status, create_dt    | 문의 데이터 조회
 문의 댓글                 | inquiry_comment                    | id, inquiry_id, content, author_id, is_admin        | 운영자 답변 조회
──────────────────────────────────────────────────────────────────

새 Tool 추가 방법:
  1. inquiry_agent.py의 InquiryLabel에 라벨 추가
  2. AUTO_TOOLS 또는 APPROVAL_TOOLS에 라벨 등록
  3. _do_xxx 내부 함수 추가
  4. create_langchain_tools()에 tool 등록
  5. execute_tool_action()에 처리 분기 추가 (하위 호환)
  6. knowledge_base.json label_examples에 예시 추가
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

from user_db import UserContextDB, CODE_REVIEW_DAILY_LIMIT, PRACTICE_DEFAULT_RESTORE

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────
# Tool 그룹 정의  (InquiryLabel 값을 문자열로 관리 — 순환 import 방지)
# ──────────────────────────────────────────────────────────────────

# AUTO TOOL: 즉시 실행 (저위험, 단순 복구)
AUTO_TOOL_LABELS: set[str] = {
    "CODE_REVIEW_RESET",
    "LITERACY_PRACTICE_RESET",
}

# APPROVAL TOOL: 관리자 승인 후 실행 (추후 추가)
APPROVAL_TOOL_LABELS: set[str] = set()

GROUP3_LABELS: set[str] = AUTO_TOOL_LABELS | APPROVAL_TOOL_LABELS


def get_tool_type(label_value: str) -> str:
    """라벨 값을 받아 'auto' 또는 'approval'을 반환."""
    if label_value in AUTO_TOOL_LABELS:
        return "auto"
    if label_value in APPROVAL_TOOL_LABELS:
        return "approval"
    return "unknown"


# ──────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────

def extract_count_from_text(text: str) -> int:
    """
    문의 텍스트에서 복구 요청 횟수를 추출.
    명시되지 않으면 PRACTICE_DEFAULT_RESTORE 반환.
    """
    m = re.search(r'(\d+)\s*(?:번|회|개|times?)', text)
    return int(m.group(1)) if m else PRACTICE_DEFAULT_RESTORE


# ──────────────────────────────────────────────────────────────────
# 내부 실행 함수 (LangChain tool & execute_tool_action 양쪽에서 호출)
#
# 운영 DB 매핑:
#   _do_code_review_reset  → code_review_log 테이블
#     - 당일 사용량: SELECT COUNT(*) FROM code_review_log
#                    WHERE user_id=? AND DATE(started_at)=CURDATE()
#                    AND status != 'FAILED'
#     - 리셋: 운영에서는 별도 리셋 플래그 또는 started_at 기준 오프셋 처리
#
#   _do_practice_reset     → literacy_test_user_status 테이블
#     - 조회: SELECT tutorial_attempts, tutorial_used
#             FROM literacy_test_user_status
#             WHERE user_id=? AND literacy_test_id=?
#     - 복구: UPDATE literacy_test_user_status
#             SET tutorial_used = GREATEST(0, tutorial_used - ?)
#             WHERE user_id=? AND literacy_test_id=?
#     - 또는 user_tutorial_status 테이블 (글로벌 사전연습)
# ──────────────────────────────────────────────────────────────────

def _do_code_review_reset(user_db: UserContextDB, user_id: int) -> Dict:
    """
    코드 리뷰 횟수 초기화 실행.

    코드 리뷰는 AI Bootcamp 최종과제 내부에 위치 (UI: "오늘 10회 남음")

    흐름 (운영 DB 기준):
      1. inquiry.author_id → user.id 로 사용자 특정
      2. 사용자의 AI Bootcamp 최종과제 lecture_id 조회
         (운영: program_final_project.lecture_ids / PoC: enrollments → cohorts 추론)
      3. code_review_log에서 해당 user_id + lecture_id 당일 카운트 리셋

    운영 테이블: code_review_log
      - user_id (int)        : 사용자 PK (= inquiry.author_id)
      - lecture_id (int)     : 최종과제 강의 ID (program_final_project에서 조회)
      - started_at (datetime): 리뷰 시작 시각 (당일 집계 기준)
      - status (enum)        : IN_PROGRESS / COMPLETED / FAILED
      - 일일 한도: 10회
    """
    user_identifier = user_db.get_user_identifier(user_id) or f"user_{user_id}"

    # 최종과제 lecture_id 조회
    # 운영: program_final_project → lecture_ids
    final_lecture_id = user_db.get_final_project_lecture_id(user_id)

    # 해당 최종과제의 코드 리뷰 횟수 리셋
    result = user_db.reset_review_count(user_id, lecture_id=final_lecture_id)
    result["user_identifier"] = user_identifier
    result["final_lecture_id"] = final_lecture_id

    if not result.get("success"):
        reason = result.get("reason", "알 수 없는 오류")
        logger.warning(
            "[tool] code_review_reset FAILED: user_id=%s (%s), final_lecture_id=%s, reason=%s",
            user_id, user_identifier, final_lecture_id, reason,
        )
        answer = (
            f"안녕하세요, AI Talent Lab입니다.\n"
            f"코드 리뷰 횟수가 아직 남아있어 초기화가 필요하지 않습니다. "
            f"현재 {result.get('remaining', CODE_REVIEW_DAILY_LIMIT)}회 사용 가능합니다.\n\n감사합니다."
        )
        return {"answer": answer, "result": result, "success": False}

    logger.info(
        "[tool] code_review_reset: user_id=%s (%s), final_lecture_id=%s, prev_used=%s",
        user_id, user_identifier, final_lecture_id, result.get("prev_used", 0),
    )

    answer = (
        f"안녕하세요, AI Talent Lab입니다.\n"
        f"코드 리뷰 횟수를 초기화했습니다. "
        f"오늘 다시 {CODE_REVIEW_DAILY_LIMIT}회 요청하실 수 있습니다.\n\n감사합니다."
    )
    return {"answer": answer, "result": result, "success": True}


def _do_practice_reset(user_db: UserContextDB, user_id: int, count: int) -> Dict:
    """
    AI Literacy 사전연습 횟수 복구 실행.

    사전연습은 AI Literacy 인증시험 내부에 위치 (UI: "인증시험 사전 연습 (97회 남음)")

    흐름 (운영 DB 기준):
      1. inquiry.author_id → user.id 로 사용자 특정
      2. 사용자의 활성 literacy_test_id 조회
         (운영: literacy_test_user_status / PoC: enrollments → AI Literacy 기수 추론)
      3. tutorial_attempts 증가 (= 복구) → 남은 횟수 = attempts - used

    운영 테이블: literacy_test_user_status
      - user_id (int)              : 사용자 PK (= inquiry.author_id)
      - literacy_test_id (bigint)  : 시험 ID (literacy_test.id)
      - tutorial_attempts (int)    : 부여된 사전연습 총 횟수
      - tutorial_used (int)        : 사용한 사전연습 횟수
      - 남은 횟수 = tutorial_attempts - tutorial_used
    """
    user_identifier = user_db.get_user_identifier(user_id) or f"user_{user_id}"

    # 사용자의 활성 인증시험 ID 조회
    # 운영: literacy_test_user_status → literacy_test.id
    # 조회 실패 시 literacy_test_id=None(글로벌)로 fallback하여 복구 진행
    literacy_test_id = user_db.get_active_literacy_test_id(user_id)

    result = user_db.restore_practice_count(user_id, count, literacy_test_id=literacy_test_id)
    result["user_identifier"] = user_identifier

    if not result.get("success"):
        reason = result.get("reason", "알 수 없는 오류")
        logger.warning(
            "[tool] practice_reset FAILED: user_id=%s (%s), literacy_test_id=%s, reason=%s",
            user_id, user_identifier, literacy_test_id, reason,
        )
        answer = (
            f"안녕하세요, AI Talent Lab입니다.\n"
            f"사전연습 횟수 복구를 처리할 수 없습니다. "
            f"담당자에게 문의해 주세요.\n\n감사합니다."
        )
        return {"answer": answer, "result": result, "success": False}

    actual_restored = result.get("restored", count)
    logger.info(
        "[tool] practice_reset: user_id=%s (%s), literacy_test_id=%s, "
        "requested=%d, restored=%d, remaining=%s",
        user_id, user_identifier, literacy_test_id,
        count, actual_restored, result.get("remaining", "?"),
    )

    answer = (
        f"안녕하세요, AI Talent Lab입니다.\n"
        f"AI Literacy 사전연습 횟수 {actual_restored}회를 복구했습니다. "
        f"다시 연습하실 수 있습니다.\n\n감사합니다."
    )
    return {"answer": answer, "result": result, "success": True}


# ──────────────────────────────────────────────────────────────────
# LangChain Tool 팩토리 (실서비스 패턴)
#
# 실서비스(code_review_agent/tools.py)의 create_langchain_tools()와 동일 패턴:
#   - 클로저로 user_db, author_id 바인딩 → LLM은 인자 없이 호출
#   - @tool 데코레이터로 LangChain 호환
#   - tool.name / tool.description 으로 LLM에 노출할 이름·설명 지정
# ──────────────────────────────────────────────────────────────────

def create_langchain_tools(
    user_db: UserContextDB,
    author_id: Optional[int],
    inquiry_text: str = "",
) -> List:
    """
    문의 처리용 LangChain tool 인스턴스 생성.

    Parameters
    ----------
    user_db      : UserContextDB 인스턴스
    author_id    : 문의자 user id (없으면 None)
    inquiry_text : 제목 + 본문 평문 (횟수 추출용)

    Returns
    -------
    list of LangChain tool instances
    """
    from langchain_core.tools import tool  # lazy import (초기 로딩 속도 개선)

    tools = []

    # ── 코드 리뷰 횟수 초기화 tool ──────────────────────────────
    # 운영 DB: code_review_log (user_id, lecture_id, started_at, status)
    def make_code_review_reset_tool(db: UserContextDB, uid: Optional[int]):
        @tool
        def reset_code_review_count() -> str:
            """수강생의 당일 코드 리뷰 사용 횟수를 초기화합니다.
            초기화 후 일일 한도만큼 다시 코드 리뷰를 요청할 수 있습니다."""
            if not uid:
                return "요청을 처리하려면 로그인 정보가 필요합니다. 담당자에게 문의해 주세요."
            try:
                out = _do_code_review_reset(db, uid)
                logger.info("[tool] reset_code_review_count: user_id=%s result=%s", uid, out["result"])
                return out["answer"]
            except Exception as e:
                logger.error("[tool] reset_code_review_count failed: user_id=%s error=%s", uid, e, exc_info=True)
                return f"코드 리뷰 횟수 초기화 중 오류가 발생했습니다: {e}"

        reset_code_review_count.name = "reset_code_review_count"
        reset_code_review_count.description = (
            "수강생의 당일 코드 리뷰 사용 횟수를 0으로 초기화합니다. "
            f"초기화 후 일일 한도({CODE_REVIEW_DAILY_LIMIT}회)만큼 다시 요청할 수 있습니다."
        )
        return reset_code_review_count

    # ── AI Literacy 사전연습 횟수 복구 tool ──────────────────────
    # 운영 DB: literacy_test_user_status (user_id, literacy_test_id,
    #          tutorial_attempts, tutorial_used)
    #   또는  user_tutorial_status (user_id, tutorial_attempts, tutorial_used)
    def make_practice_reset_tool(db: UserContextDB, uid: Optional[int], text: str):
        @tool
        def restore_practice_count() -> str:
            """수강생의 AI Literacy 사전연습 횟수를 복구(추가 부여)합니다.
            문의 내용에서 요청 횟수를 추출하여 해당 횟수만큼 복구합니다."""
            if not uid:
                return "요청을 처리하려면 로그인 정보가 필요합니다. 담당자에게 문의해 주세요."
            try:
                count = extract_count_from_text(text)
                out = _do_practice_reset(db, uid, count)
                logger.info("[tool] restore_practice_count: user_id=%s count=%d result=%s", uid, count, out["result"])
                return out["answer"]
            except Exception as e:
                logger.error("[tool] restore_practice_count failed: user_id=%s error=%s", uid, e, exc_info=True)
                return f"사전연습 횟수 복구 중 오류가 발생했습니다: {e}"

        restore_practice_count.name = "restore_practice_count"
        restore_practice_count.description = (
            "수강생의 AI Literacy 사전연습 횟수를 복구합니다. "
            f"기본 {PRACTICE_DEFAULT_RESTORE}회 복구되며, 문의 내용에 횟수가 명시되면 해당 횟수만큼 복구합니다."
        )
        return restore_practice_count

    # ── tool 인스턴스 생성 (클로저로 context 바인딩) ──────────────
    tools.append(make_code_review_reset_tool(user_db, author_id))
    tools.append(make_practice_reset_tool(user_db, author_id, inquiry_text))

    # ── APPROVAL TOOLS (추후 추가 시 여기에 등록) ────────────────

    return tools


# ──────────────────────────────────────────────────────────────────
# Tool 실행기 (하위 호환 — inquiry_agent.py에서 직접 호출)
#
# inquiry_agent.py의 기존 호출 방식을 유지:
#   answer_text, tool_result, tool_type = execute_tool_action(
#       label_value, author_id, inquiry_text, user_db
#   )
# ──────────────────────────────────────────────────────────────────

def execute_tool_action(
    label_value: str,
    author_id: Optional[int],
    inquiry_text: str,
    user_db: UserContextDB,
) -> Tuple[str, Dict, str]:
    """
    GROUP3 라벨에 대응하는 실제 액션을 실행.

    Parameters
    ----------
    label_value  : InquiryLabel.value 문자열
    author_id    : 문의자 user id (없으면 None)
    inquiry_text : 제목 + 본문 평문 (횟수 추출용)
    user_db      : UserContextDB 인스턴스

    Returns
    -------
    (answer_text, tool_result, tool_type)
    """
    tool_type = get_tool_type(label_value)

    if not author_id:
        answer = (
            "안녕하세요, AI Talent Lab입니다.\n"
            "요청을 처리하려면 로그인 정보가 필요합니다. 담당자에게 문의해 주세요.\n\n감사합니다."
        )
        return answer, {"success": False, "reason": "author_id 없음"}, tool_type

    # ── AUTO TOOLS ─────────────────────────────────────────────────

    if label_value == "CODE_REVIEW_RESET":
        out = _do_code_review_reset(user_db, author_id)
        return out["answer"], out["result"], tool_type

    if label_value == "LITERACY_PRACTICE_RESET":
        count = extract_count_from_text(inquiry_text)
        out = _do_practice_reset(user_db, author_id, count)
        return out["answer"], out["result"], tool_type

    # ── APPROVAL TOOLS (추후 추가) ─────────────────────────────────

    # 알 수 없는 라벨 (방어 코드)
    return (
        "안녕하세요, AI Talent Lab입니다.\n요청을 처리할 수 없습니다. 담당자에게 문의해 주세요.\n\n감사합니다.",
        {"success": False, "reason": f"미지원 액션: {label_value}"},
        tool_type,
    )
