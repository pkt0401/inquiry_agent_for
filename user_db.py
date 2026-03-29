"""
user_db.py — 수강생 개인화 데이터 (SQLite)

테이블:
  users       : 수강생 기본 정보
  cohorts     : 기수 정보 (프로그램별)
  enrollments : 수강 이력 (기수별 수료/미수료/진행중)

PoC 용 더미 데이터를 포함.
실서비스에서는 PostgreSQL 등으로 교체 가능 (SQLAlchemy 기반으로 변경하면 됨).
"""

import sqlite3
import os
import random
from typing import Dict, List, Optional


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")


# ──────────────────────────────────────────────────────────────────
# DB 초기화 / 스키마
# ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    email       TEXT
);

CREATE TABLE IF NOT EXISTS cohorts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    program         TEXT NOT NULL,   -- 'AI Bootcamp' | 'AI Literacy' | 'AI Master Project'
    cohort_name     TEXT NOT NULL,   -- '10기' | '11기' | '12기' 등
    start_dt        TEXT,
    end_dt          TEXT
);

CREATE TABLE IF NOT EXISTS enrollments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    cohort_id   INTEGER NOT NULL,
    status      TEXT NOT NULL,   -- 'completed' | 'failed' | 'in_progress'
    final_score REAL,
    note        TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (cohort_id) REFERENCES cohorts(id)
);

-- 운영 DB: code_review_log (user_id, lecture_id, started_at, status)
-- PoC: 집계형으로 단순화. lecture_id 추가하여 강의별 한도 관리.
CREATE TABLE IF NOT EXISTS code_review_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    lecture_id  INTEGER,                -- 강의 ID (NULL이면 전체 리셋)
    review_dt   TEXT NOT NULL,          -- 'YYYY-MM-DD' (날짜)
    used_count  INTEGER NOT NULL DEFAULT 0,
    reset_count INTEGER NOT NULL DEFAULT 0,   -- 당일 리셋 횟수
    last_reset_at TEXT,                        -- 마지막 리셋 시각 (ISO 8601)
    UNIQUE (user_id, lecture_id, review_dt)
);

-- 운영 DB: literacy_test_user_status (user_id, literacy_test_id, tutorial_attempts, tutorial_used)
--       또는 user_tutorial_status (user_id, tutorial_attempts, tutorial_used)
-- PoC: literacy_test_id 추가하여 시험별 관리 가능하도록 변경.
CREATE TABLE IF NOT EXISTS practice_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,
    literacy_test_id    INTEGER,                -- 시험 ID (NULL이면 글로벌)
    tutorial_attempts   INTEGER NOT NULL DEFAULT 0,   -- 부여된 연습 총 횟수 (= 운영 tutorial_attempts)
    tutorial_used       INTEGER NOT NULL DEFAULT 0,   -- 사용한 연습 횟수 (= 운영 tutorial_used)
    last_restore_at     TEXT,                          -- 마지막 복구 시각 (ISO 8601)
    UNIQUE (user_id, literacy_test_id)
);
"""

CODE_REVIEW_DAILY_LIMIT = 10
PRACTICE_INITIAL_COUNT   = 100  # 사전연습 초기 부여 횟수
PRACTICE_DEFAULT_RESTORE = 1    # 복구 요청 시 기본 복구 횟수 (잘못 클릭 1회 원복)

STATUS_KO = {
    "completed":   "수료",
    "failed":      "미수료",
    "in_progress": "진행중",
}

_STATUS_I18N = {
    'en': {"completed": "Completed", "failed": "Failed", "in_progress": "In Progress"},
    'jp': {"completed": "修了", "failed": "未修了", "in_progress": "受講中"},
}

_CTX_I18N = {
    'ko': dict(
        header="## 문의자 수강 이력",
        current="- 현재 과정",
        retake="재수강",
        retake_note="미수료 이력 있음",
        history="- 전체 이력:",
        score="점수",
    ),
    'en': dict(
        header="## Inquiry Author Enrollment History",
        current="- Current Course",
        retake="Re-enrollment",
        retake_note="has prior incomplete record",
        history="- Full History:",
        score="score",
    ),
    'jp': dict(
        header="## 問い合わせ者の受講履歴",
        current="- 現在のコース",
        retake="再受講",
        retake_note="未修了の履歴あり",
        history="- 全履歴:",
        score="スコア",
    ),
}


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH):
    """DB 파일이 없으면 스키마 생성 + 더미 데이터 삽입."""
    exists = os.path.exists(db_path)
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    if not exists:
        _insert_dummy_data(conn)
    conn.close()


# ──────────────────────────────────────────────────────────────────
# 더미 데이터
# ──────────────────────────────────────────────────────────────────

# inquiry_all.json 의 실제 author_id 목록 (110건 전체 기준)
KNOWN_AUTHOR_IDS = [
    2, 277, 312, 400, 404, 752, 1133, 1194, 1256, 1306,
    1467, 1512, 1792, 1846, 1882, 1910, 2045, 2343, 2384, 2607,
    2656, 2710, 2844, 2860, 2894, 2969, 2985, 2996, 3010, 3023,
    3346, 3405, 3406, 3417, 3508, 3518, 3571, 3600, 3621, 3627,
    3646, 3652, 3658, 3691, 3710, 3890, 4011, 4100, 4183, 4358,
    4403, 4560, 4586, 7918, 7938, 8137, 8193, 8540, 8660, 11262,
    11306, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010,
    20011, 20012, 20013, 20014, 20015, 20016, 20017, 20020, 20021, 20022,
    20024, 20026, 20027, 20028, 20029, 20063,
]

COHORT_DATA = [
    # (program, cohort_name, start_dt, end_dt)
    ("AI Bootcamp", "10기", "2025-12-25", "2026-01-09"),
    ("AI Bootcamp", "11기", "2026-01-19", "2026-02-13"),
    ("AI Bootcamp", "12기", "2026-02-23", "2026-03-20"),   # 현재 진행 중 (schedule.json 기준)
    ("AI Literacy", "상시", "2024-01-01", "2099-12-31"),    # 상시 운영
    ("AI Master Project", "4기", "2026-01-26", "2026-03-27"),
]


def _insert_dummy_data(conn: sqlite3.Connection):
    random.seed(42)

    # cohorts
    cohort_ids = {}
    for program, cohort_name, start_dt, end_dt in COHORT_DATA:
        cur = conn.execute(
            "INSERT INTO cohorts (program, cohort_name, start_dt, end_dt) VALUES (?,?,?,?)",
            (program, cohort_name, start_dt, end_dt),
        )
        cohort_ids[(program, cohort_name)] = cur.lastrowid
    conn.commit()

    bc_12  = cohort_ids[("AI Bootcamp", "12기")]
    bc_11  = cohort_ids[("AI Bootcamp", "11기")]
    bc_10  = cohort_ids[("AI Bootcamp", "10기")]
    lit    = cohort_ids[("AI Literacy", "상시")]

    for uid in KNOWN_AUTHOR_IDS:
        conn.execute("INSERT OR IGNORE INTO users (id, name, email) VALUES (?,?,?)",
                     (uid, f"user_{uid}", f"user_{uid}@example.com"))

        r = random.random()

        if r < 0.12:
            # 케이스 A: 10기 미수료 → 12기 재수강
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_10, "failed", round(random.uniform(30, 59), 1), "최종과제 미제출로 미수료"),
            )
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_12, "in_progress", None, "재수강"),
            )

        elif r < 0.22:
            # 케이스 B: 11기 미수료 → 12기 재수강
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_11, "failed", round(random.uniform(20, 59), 1), "최종과제 점수 미달"),
            )
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_12, "in_progress", None, "재수강"),
            )

        elif r < 0.50:
            # 케이스 C: 12기 신규 수강중
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_12, "in_progress", None, None),
            )

        elif r < 0.65:
            # 케이스 D: AI Literacy 수강 중
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, lit, "in_progress", None, None),
            )

        elif r < 0.80:
            # 케이스 E: AI Literacy 수강 중 + Bootcamp 12기 진행중
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, lit, "in_progress", None, None),
            )
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_12, "in_progress", None, None),
            )

        elif r < 0.88:
            # 케이스 F: Bootcamp 11기 수료 + AI Literacy 수료
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, bc_11, "completed", round(random.uniform(70, 100), 1), None),
            )
            conn.execute(
                "INSERT INTO enrollments (user_id, cohort_id, status, final_score, note) VALUES (?,?,?,?,?)",
                (uid, lit, "completed", None, "인증시험 합격"),
            )

        # else: 수강 이력 없음

    conn.commit()

    # ── 코드 리뷰 / 사전연습 사용 기록 (TOOL_ACTION 테스트용) ────────
    import datetime
    today = datetime.date.today().isoformat()

    # 코드 리뷰 일일 한도(10회)를 다 쓴 사용자들
    review_heavy_users = {
        2343: 10,   # AI Bootcamp 12기 재수강, 한도 전부 소진
        3010: 10,   # AI Bootcamp 12기, 한도 전부 소진
        3405: 9,    # 거의 다 씀
        3518: 10,   # 한도 전부 소진
        1792: 8,    # 많이 씀
    }
    for uid, used in review_heavy_users.items():
        conn.execute(
            "INSERT OR IGNORE INTO code_review_logs "
            "(user_id, review_dt, used_count, reset_count) VALUES (?,?,?,0)",
            (uid, today, used),
        )

    # 사전연습 횟수를 사용/소진한 사용자들 (tutorial_attempts=부여, tutorial_used=사용)
    practice_users_data = [
        (752,  100, 98),   # AI Literacy 수강중, 거의 다 씀 (2회 남음)
        (1133, 100, 100),  # 전부 소진 (0회 남음)
        (3010, 100, 45),   # 일부 사용
        (1306, 100, 99),   # 거의 다 씀 (1회 남음)
    ]
    for uid, attempts, used in practice_users_data:
        # literacy_test_id는 get_active_literacy_test_id()에서 반환하는 값과 매칭
        lit_test_id = None
        lit_row = conn.execute(
            "SELECT c.id as cohort_id FROM enrollments e "
            "JOIN cohorts c ON e.cohort_id = c.id "
            "WHERE e.user_id = ? AND c.program = 'AI Literacy' LIMIT 1",
            (uid,),
        ).fetchone()
        if lit_row:
            lit_test_id = lit_row["cohort_id"] * 10 + 1
        conn.execute(
            "INSERT OR IGNORE INTO practice_sessions "
            "(user_id, literacy_test_id, tutorial_attempts, tutorial_used) VALUES (?,?,?,?)",
            (uid, lit_test_id, attempts, used),
        )

    conn.commit()
    print(f"[user_db] 더미 데이터 삽입 완료: {len(KNOWN_AUTHOR_IDS)}명 "
          f"(코드리뷰 {len(review_heavy_users)}건, 사전연습 {len(practice_users_data)}건)")


# ──────────────────────────────────────────────────────────────────
# 조회 API
# ──────────────────────────────────────────────────────────────────

class UserContextDB:
    """수강생 개인 맥락 조회 클래스. inquiry_agent.py 에서 사용."""

    def __init__(self, db_path: str = DB_PATH):
        init_db(db_path)
        self._db_path = db_path

    def get_user_context(self, author_id: int) -> Optional[Dict]:
        """
        author_id → 수강 이력 딕셔너리 반환.
        수강 이력이 없으면 None 반환.

        반환 예:
        {
            "user_id": 1256,
            "enrollments": [
                {"program": "AI Bootcamp", "cohort": "10기",
                 "status": "failed", "status_ko": "미수료",
                 "final_score": 45.0, "note": "최종과제 미제출로 미수료"},
                {"program": "AI Bootcamp", "cohort": "12기",
                 "status": "in_progress", "status_ko": "진행중",
                 "final_score": None, "note": "재수강"},
            ],
            "current_program": "AI Bootcamp",
            "current_cohort": "12기",
            "is_retake": True,
            "retake_from": "10기",
        }
        """
        if not author_id:
            return None

        conn = get_connection(self._db_path)
        rows = conn.execute(
            """
            SELECT c.program, c.cohort_name, c.start_dt, c.end_dt,
                   e.status, e.final_score, e.note
            FROM enrollments e
            JOIN cohorts c ON e.cohort_id = c.id
            WHERE e.user_id = ?
            ORDER BY c.start_dt
            """,
            (author_id,),
        ).fetchall()
        conn.close()

        if not rows:
            return None

        enrollments = [
            {
                "program":     r["program"],
                "cohort":      r["cohort_name"],
                "start_dt":    r["start_dt"],
                "end_dt":      r["end_dt"],
                "status":      r["status"],
                "status_ko":   STATUS_KO.get(r["status"], r["status"]),
                "final_score": r["final_score"],
                "note":        r["note"],
            }
            for r in rows
        ]

        # 현재 수강 중인 과정
        in_progress = [e for e in enrollments if e["status"] == "in_progress"]
        current = in_progress[-1] if in_progress else enrollments[-1]

        # 재수강 여부: 같은 프로그램에 failed 이력이 있으면 재수강
        failed = [e for e in enrollments if e["status"] == "failed"
                  and e["program"] == current["program"]]
        is_retake = len(failed) > 0
        retake_from = failed[-1]["cohort"] if is_retake else None

        return {
            "user_id":         author_id,
            "enrollments":     enrollments,
            "current_program": current["program"],
            "current_cohort":  current["cohort"],
            "is_retake":       is_retake,
            "retake_from":     retake_from,
        }

    # ── 코드 리뷰 대상 조회: 최종과제 lecture_id ──────────────────────
    #
    # 코드 리뷰는 AI Bootcamp 최종과제 안에 있음 (UI: "오늘 10회 남음")
    # → 사용자의 현재 진행 중인 Bootcamp 기수 → 최종과제 lecture_id 특정 필요
    #
    # 운영 DB 조회 흐름:
    #   user_cohorts (user_id, cohort_id)
    #     → cohort (id, program_id) WHERE program.type = 'bootcamp'
    #     → program_final_project (program_id → lecture_ids)
    #     → lecture (id) ← 최종과제 강의
    #   또는 직접: code_review_log에서 해당 user_id의 최근 lecture_id 조회

    def get_final_project_lecture_id(self, user_id: int) -> Optional[int]:
        """
        user_id의 현재 진행 중인 AI Bootcamp 최종과제 lecture_id를 반환.

        운영 DB:
          SELECT pfp.lecture_ids
          FROM user_cohorts uc
          JOIN cohort c ON uc.cohort_id = c.id
          JOIN program_final_project pfp ON pfp.program_id = c.program_id
          WHERE uc.user_id = :user_id
          ORDER BY c.start_date DESC LIMIT 1

        PoC: enrollments → AI Bootcamp 진행중 → cohort_id 기반 더미 lecture_id
        """
        conn = get_connection(self._db_path)
        row = conn.execute(
            """
            SELECT c.id as cohort_id, c.cohort_name
            FROM enrollments e
            JOIN cohorts c ON e.cohort_id = c.id
            WHERE e.user_id = ? AND e.status = 'in_progress'
              AND c.program = 'AI Bootcamp'
            ORDER BY c.start_dt DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None
        # PoC: cohort_id 기반 더미 lecture_id (실서비스에서는 program_final_project.lecture_ids)
        return row["cohort_id"] * 100 + 1

    # ── 사전연습 대상 조회: literacy_test_id ──────────────────────────
    #
    # 사전연습은 AI Literacy 인증시험 안에 있음 (UI: "인증시험 사전 연습 (97회 남음)")
    # → 사용자의 현재 해당하는 literacy_test_id 특정 필요
    #
    # 운영 DB 조회 흐름:
    #   literacy_test_user_status (user_id, literacy_test_id, tutorial_attempts, tutorial_used)
    #   또는:
    #     user_cohorts (user_id, cohort_id)
    #       → literacy_test (cohort_id → id)
    #       → literacy_test_user_status (user_id, literacy_test_id)

    def get_active_literacy_test_id(self, user_id: int) -> Optional[int]:
        """
        user_id의 현재 활성화된 literacy_test_id를 반환.

        운영 DB:
          SELECT ltus.literacy_test_id
          FROM literacy_test_user_status ltus
          JOIN literacy_test lt ON ltus.literacy_test_id = lt.id
          WHERE ltus.user_id = :user_id
            AND lt.is_active = 1  -- 또는 test_end_time > NOW()
          ORDER BY lt.test_start_time DESC LIMIT 1

        PoC: AI Literacy 수강 중이면 더미 literacy_test_id 반환
        """
        conn = get_connection(self._db_path)
        row = conn.execute(
            """
            SELECT c.id as cohort_id
            FROM enrollments e
            JOIN cohorts c ON e.cohort_id = c.id
            WHERE e.user_id = ? AND e.status = 'in_progress'
              AND c.program = 'AI Literacy'
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None
        # PoC: cohort_id 기반 더미 test_id (실서비스에서는 literacy_test.id)
        return row["cohort_id"] * 10 + 1

    def get_user_identifier(self, user_id: int) -> Optional[str]:
        """
        user_id → user_identifier(사번/로그인 ID) 반환.

        운영 DB: SELECT user_id FROM user WHERE id = ?
        PoC: email에서 추출.
        """
        conn = get_connection(self._db_path)
        row = conn.execute("SELECT email FROM users WHERE id=?", (user_id,)).fetchone()
        conn.close()
        if not row:
            return None
        return row["email"].split("@")[0] if row["email"] else f"user_{user_id}"

    # ── 코드 리뷰 횟수 관리 ─────────────────────────────────────────
    #
    # 운영 DB 흐름:
    #   1. inquiry.author_id → user.id 로 사용자 특정
    #   2. user_lecture 또는 enrollments → 현재 수강 중인 lecture_id 목록
    #   3. code_review_log에서 해당 user_id + lecture_id 당일 카운트 조회
    #   4. 리셋: 해당 레코드 used_count = 0 또는 리셋 플래그 처리
    #

    def get_review_count_today(self, user_id: int, lecture_id: int = None, date_str: str = None) -> Dict:
        """
        user_id의 오늘 코드 리뷰 사용 현황 반환.

        Parameters
        ----------
        user_id    : 사용자 PK (= inquiry.author_id)
        lecture_id : 강의 ID (None이면 전체 합산)
        date_str   : 'YYYY-MM-DD' (기본: 오늘)

        운영 DB 쿼리:
          SELECT COUNT(*) FROM code_review_log
          WHERE user_id = :user_id
            AND lecture_id = :lecture_id
            AND DATE(started_at) = :date
            AND status != 'FAILED'
        """
        import datetime
        if date_str is None:
            date_str = datetime.date.today().isoformat()

        conn = get_connection(self._db_path)
        if lecture_id is not None:
            row = conn.execute(
                "SELECT used_count, reset_count FROM code_review_logs "
                "WHERE user_id=? AND lecture_id=? AND review_dt=?",
                (user_id, lecture_id, date_str),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT SUM(used_count) as used_count, SUM(reset_count) as reset_count "
                "FROM code_review_logs WHERE user_id=? AND review_dt=?",
                (user_id, date_str),
            ).fetchone()
        conn.close()

        used = row["used_count"] if row and row["used_count"] else 0
        reset_cnt = row["reset_count"] if row and row["reset_count"] else 0
        return {
            "user_id":     user_id,
            "lecture_id":  lecture_id,
            "date":        date_str,
            "used_count":  used,
            "remaining":   max(0, CODE_REVIEW_DAILY_LIMIT - used),
            "daily_limit": CODE_REVIEW_DAILY_LIMIT,
            "reset_count": reset_cnt,
        }

    def reset_review_count(self, user_id: int, lecture_id: int = None, date_str: str = None) -> Dict:
        """
        user_id의 당일 코드 리뷰 사용 횟수를 0으로 초기화.

        Parameters
        ----------
        user_id    : 사용자 PK (= inquiry.author_id → user.id)
        lecture_id : 강의 ID (None이면 해당 유저의 당일 전체 리셋)
        date_str   : 'YYYY-MM-DD' (기본: 오늘)

        운영 DB 동작:
          1. user.id = inquiry.author_id 로 사용자 특정
          2. code_review_log에서 해당 user_id + lecture_id + 당일 레코드 리셋
        """
        import datetime
        if date_str is None:
            date_str = datetime.date.today().isoformat()
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        conn = get_connection(self._db_path)

        if lecture_id is not None:
            row = conn.execute(
                "SELECT used_count, reset_count FROM code_review_logs "
                "WHERE user_id=? AND lecture_id=? AND review_dt=?",
                (user_id, lecture_id, date_str),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT SUM(used_count) as used_count, SUM(reset_count) as reset_count "
                "FROM code_review_logs WHERE user_id=? AND review_dt=?",
                (user_id, date_str),
            ).fetchone()

        # 레코드 없음 = 오늘 한 번도 안 씀 = 10회 전부 남음 → 리셋 불필요
        prev_used = row["used_count"] if row and row["used_count"] else 0
        if prev_used == 0:
            conn.close()
            return {
                "success":     False,
                "user_id":     user_id,
                "lecture_id":  lecture_id,
                "date":        date_str,
                "prev_used":   0,
                "reset_count": 0,
                "remaining":   CODE_REVIEW_DAILY_LIMIT,
                "reason":      f"오늘 사용 이력이 없습니다 (잔여 {CODE_REVIEW_DAILY_LIMIT}회)",
            }

        if lecture_id is not None:
            conn.execute(
                """
                UPDATE code_review_logs
                SET used_count = 0, reset_count = reset_count + 1, last_reset_at = ?
                WHERE user_id = ? AND lecture_id = ? AND review_dt = ?
                """,
                (now_iso, user_id, lecture_id, date_str),
            )
            new_reset_cnt = (row["reset_count"] if row else 0) + 1
        else:
            conn.execute(
                "UPDATE code_review_logs SET used_count=0, reset_count=reset_count+1, "
                "last_reset_at=? WHERE user_id=? AND review_dt=?",
                (now_iso, user_id, date_str),
            )
            new_reset_cnt = -1  # 여러 행 업데이트됨

        conn.commit()
        conn.close()

        return {
            "success":     True,
            "user_id":     user_id,
            "lecture_id":  lecture_id,
            "date":        date_str,
            "prev_used":   prev_used,
            "reset_count": new_reset_cnt,
        }

    # ── AI Literacy 사전연습 횟수 관리 ──────────────────────────────
    #
    # 운영 DB 흐름:
    #   1. inquiry.author_id → user.id 로 사용자 특정
    #   2. literacy_test_user_status에서 해당 user_id + literacy_test_id 조회
    #   3. tutorial_used 차감 또는 tutorial_attempts 증가
    #

    def restore_practice_count(self, user_id: int, count: int = PRACTICE_DEFAULT_RESTORE,
                                literacy_test_id: int = None) -> Dict:
        """
        user_id의 AI Literacy 사전연습 횟수를 count만큼 복구.
        실제 운영: 잘못 차감된 tutorial_used를 줄여주는 방식.
        (운영자가 DB에서 그날 시도 기록을 전날로 옮기는 것과 동일 효과)

        Parameters
        ----------
        user_id          : 사용자 PK (= inquiry.author_id → user.id)
        count            : 복구할 횟수
        literacy_test_id : 시험 ID (None이면 글로벌 레코드로 처리)
        """
        import datetime
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        conn = get_connection(self._db_path)

        # literacy_test_id가 None이면 IS NULL 조건 사용 (SQL에서 NULL=NULL은 False)
        if literacy_test_id is not None:
            _where = "user_id=? AND literacy_test_id=?"
            _params = (user_id, literacy_test_id)
        else:
            _where = "user_id=? AND literacy_test_id IS NULL"
            _params = (user_id,)

        # 현재 상태 조회
        row = conn.execute(
            f"SELECT tutorial_attempts, tutorial_used FROM practice_sessions WHERE {_where}",
            _params,
        ).fetchone()

        # 레코드 없음 → 새 레코드 생성 (수강 이력 없어도 복구 처리)
        if row is None:
            conn.execute(
                "INSERT INTO practice_sessions "
                "(user_id, literacy_test_id, tutorial_attempts, tutorial_used, last_restore_at) "
                "VALUES (?, ?, ?, 0, ?)",
                (user_id, literacy_test_id, PRACTICE_INITIAL_COUNT, now_iso),
            )
            conn.commit()
            conn.close()
            return {
                "success":            True,
                "user_id":            user_id,
                "literacy_test_id":   literacy_test_id,
                "restored":           count,
                "requested":          count,
                "tutorial_attempts":  PRACTICE_INITIAL_COUNT,
                "tutorial_used":      0,
                "remaining":          PRACTICE_INITIAL_COUNT,
                "note":               "신규 레코드 생성 (사전연습 사용 이력 없음, 복구 완료)",
            }

        current_attempts = row["tutorial_attempts"]
        current_used     = row["tutorial_used"]

        # 복구 = tutorial_used 차감 (잘못 차감된 사용 횟수를 되돌림)
        actual_count = min(count, current_used)  # 0 미만으로 내려가지 않도록

        if actual_count <= 0:
            conn.close()
            remaining = current_attempts - current_used
            return {
                "success":            True,
                "user_id":            user_id,
                "literacy_test_id":   literacy_test_id,
                "restored":           0,
                "tutorial_attempts":  current_attempts,
                "tutorial_used":      current_used,
                "remaining":          remaining,
                "note":               f"사용 횟수가 0이므로 차감할 것이 없습니다 (잔여 {remaining}회)",
            }

        conn.execute(
            f"""
            UPDATE practice_sessions
            SET tutorial_used   = tutorial_used - ?,
                last_restore_at = ?
            WHERE {_where}
            """,
            (actual_count, now_iso) + _params,
        )
        conn.commit()
        row = conn.execute(
            f"SELECT tutorial_attempts, tutorial_used FROM practice_sessions WHERE {_where}",
            _params,
        ).fetchone()
        conn.close()

        remaining = (row["tutorial_attempts"] - row["tutorial_used"]) if row else actual_count
        return {
            "success":            True,
            "user_id":            user_id,
            "literacy_test_id":   literacy_test_id,
            "restored":           actual_count,
            "requested":          count,
            "tutorial_attempts":  row["tutorial_attempts"] if row else current_attempts,
            "tutorial_used":      row["tutorial_used"] if row else 0,
            "remaining":          remaining,
        }

    def build_personal_context_str(self, author_id: int, lang: str = 'ko') -> str:
        """
        프롬프트에 주입할 수강생 개인 맥락 문자열 반환.
        수강 이력 없으면 빈 문자열 반환.
        lang: 'ko' | 'en' | 'jp'
        """
        ctx = self.get_user_context(author_id)
        if not ctx:
            return ""

        i18n = _CTX_I18N.get(lang, _CTX_I18N['ko'])
        status_map = _STATUS_I18N.get(lang, STATUS_KO)

        lines = [i18n['header']]
        in_progress_label = status_map.get('in_progress', STATUS_KO['in_progress'])
        current_line = f"{i18n['current']}: {ctx['current_program']} {ctx['current_cohort']} ({in_progress_label})"
        if ctx["is_retake"]:
            current_line += f" — {i18n['retake']} ({ctx['retake_from']} {i18n['retake_note']})"
        lines.append(current_line)

        lines.append(i18n['history'])
        for e in ctx["enrollments"]:
            status_label = status_map.get(e['status'], e['status_ko'])
            score_str = f" ({i18n['score']}: {e['final_score']})" if e["final_score"] is not None else ""
            note_str  = f" — {e['note']}" if e["note"] else ""
            lines.append(f"  · {e['program']} {e['cohort']}: {status_label}{score_str}{note_str}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# 간단 테스트
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    db = UserContextDB()
    print("=== 수강생 맥락 조회 테스트 ===\n")

    for uid in [1256, 400, 312, 9999]:
        print(f"--- author_id={uid} ---")
        ctx_str = db.build_personal_context_str(uid)
        print(ctx_str if ctx_str else "(수강 이력 없음)")
        print()
