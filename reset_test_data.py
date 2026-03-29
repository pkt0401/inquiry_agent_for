"""
reset_test_data.py — Tool 테스트용 DB 유틸리티

사용법:
  python reset_test_data.py                  # 더미 데이터 초기화 (리뷰 사용량 + 연습 사용량 세팅)
  python reset_test_data.py --show           # 현재 DB 상태만 조회
  python reset_test_data.py --use-review 277 3       # user=277의 코드 리뷰 3회 사용 추가
  python reset_test_data.py --use-practice 400 5     # user=400의 사전연습 5회 사용 추가
"""

import sqlite3
import datetime
import argparse
import os
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data.db")
CODE_REVIEW_DAILY_LIMIT = 10
PRACTICE_MAX = 100


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def show_status():
    """현재 DB 상태 출력."""
    conn = get_conn()
    today = datetime.date.today().isoformat()

    rows = conn.execute(
        "SELECT * FROM code_review_logs WHERE review_dt=? ORDER BY user_id", (today,)
    ).fetchall()

    print(f"=== code_review_logs (오늘: {today}, 일일 한도: {CODE_REVIEW_DAILY_LIMIT}회) ===")
    print(f"{'user_id':>8} {'lecture':>8} {'사용':>5} {'남은':>5} {'리셋':>5}")
    print("-" * 40)
    if not rows:
        print("  (데이터 없음)")
    for r in rows:
        rem = CODE_REVIEW_DAILY_LIMIT - r["used_count"]
        print(f"{r['user_id']:>8} {r['lecture_id']:>8} {r['used_count']:>5} {rem:>5} {r['reset_count']:>5}")

    print()
    rows = conn.execute(
        "SELECT * FROM practice_sessions ORDER BY user_id"
    ).fetchall()

    print(f"=== practice_sessions (상한: {PRACTICE_MAX}회) ===")
    print(f"{'user_id':>8} {'test_id':>8} {'부여':>5} {'사용':>5} {'남은':>5}")
    print("-" * 40)
    if not rows:
        print("  (데이터 없음)")
    for r in rows:
        rem = r["tutorial_attempts"] - r["tutorial_used"]
        print(f"{r['user_id']:>8} {r['literacy_test_id'] or 'None':>8} {r['tutorial_attempts']:>5} {r['tutorial_used']:>5} {rem:>5}")

    conn.close()


def init_dummy():
    """더미 데이터 초기화 — 코드 리뷰 사용량 + 사전연습 사용량 세팅."""
    conn = get_conn()
    today = datetime.date.today().isoformat()

    conn.execute("DELETE FROM code_review_logs")
    conn.execute("DELETE FROM practice_sessions")

    import random as _rng
    _rng.seed(42)

    # ── 코드 리뷰: AI Bootcamp 진행중인 user 전원 ──
    bc_users = [r["user_id"] for r in conn.execute("""
        SELECT e.user_id FROM enrollments e
        JOIN cohorts c ON e.cohort_id = c.id
        WHERE c.program = 'AI Bootcamp' AND e.status = 'in_progress'
    """).fetchall()]

    for uid in bc_users:
        # cohort_id 기반 더미 lecture_id (user_db.py와 동일 로직)
        cohort_row = conn.execute("""
            SELECT c.id FROM enrollments e
            JOIN cohorts c ON e.cohort_id = c.id
            WHERE e.user_id = ? AND c.program = 'AI Bootcamp' AND e.status = 'in_progress'
            ORDER BY c.start_dt DESC LIMIT 1
        """, (uid,)).fetchone()
        lecture_id = cohort_row["id"] * 100 + 1 if cohort_row else 301
        used = _rng.choice([0, 2, 3, 5, 7, 8, 9, 10, 10])  # 다양한 사용량
        conn.execute(
            "INSERT INTO code_review_logs "
            "(user_id, lecture_id, review_dt, used_count, reset_count) "
            "VALUES (?,?,?,?,0)",
            (uid, lecture_id, today, used),
        )

    # ── 사전연습: AI Literacy 수강중인 user 전원 ──
    lit_cohort_id = conn.execute(
        "SELECT id FROM cohorts WHERE program='AI Literacy'"
    ).fetchone()["id"]
    lit_test_id = lit_cohort_id * 10 + 1  # user_db.py와 동일 로직 (= 41)

    lit_users = [r["user_id"] for r in conn.execute("""
        SELECT e.user_id FROM enrollments e
        JOIN cohorts c ON e.cohort_id = c.id
        WHERE c.program = 'AI Literacy' AND e.status = 'in_progress'
    """).fetchall()]

    for uid in lit_users:
        used = _rng.choice([3, 12, 25, 45, 60, 80, 95, 98, 100])
        # 일부는 이전에 복구받은 적 있는 상태 (attempts < 100)
        if _rng.random() < 0.15:
            attempts = _rng.randint(90, 99)
            used = attempts  # 소진 상태
        else:
            attempts = PRACTICE_MAX
        conn.execute(
            "INSERT INTO practice_sessions "
            "(user_id, literacy_test_id, tutorial_attempts, tutorial_used) "
            "VALUES (?,?,?,?)",
            (uid, lit_test_id, attempts, used),
        )

    conn.commit()
    conn.close()
    print("더미 데이터 초기화 완료\n")
    show_status()


def use_review(user_id: int, count: int):
    """코드 리뷰 사용량 증가 (= 남은 횟수 감소)."""
    conn = get_conn()
    today = datetime.date.today().isoformat()

    row = conn.execute(
        "SELECT used_count FROM code_review_logs "
        "WHERE user_id=? AND review_dt=?",
        (user_id, today),
    ).fetchone()

    if not row:
        # 레코드 없으면 새로 생성
        conn.execute(
            "INSERT INTO code_review_logs "
            "(user_id, lecture_id, review_dt, used_count, reset_count) "
            "VALUES (?, 301, ?, ?, 0)",
            (user_id, today, count),
        )
        print(f"user={user_id}: 신규 생성, used=0 → {count}")
    else:
        current = row["used_count"]
        new_used = min(current + count, CODE_REVIEW_DAILY_LIMIT)
        conn.execute(
            "UPDATE code_review_logs SET used_count=? "
            "WHERE user_id=? AND review_dt=?",
            (new_used, user_id, today),
        )
        print(f"user={user_id}: used={current} → {new_used} (남은: {CODE_REVIEW_DAILY_LIMIT - new_used})")

    conn.commit()
    conn.close()


def use_practice(user_id: int, count: int):
    """사전연습 사용량 증가 (= 남은 횟수 감소)."""
    conn = get_conn()

    row = conn.execute(
        "SELECT tutorial_attempts, tutorial_used FROM practice_sessions "
        "WHERE user_id=?",
        (user_id,),
    ).fetchone()

    if not row:
        print(f"user={user_id}: practice_sessions 레코드 없음")
        conn.close()
        return

    current_used = row["tutorial_used"]
    attempts = row["tutorial_attempts"]
    new_used = min(current_used + count, attempts)

    conn.execute(
        "UPDATE practice_sessions SET tutorial_used=? WHERE user_id=?",
        (new_used, user_id),
    )
    conn.commit()
    conn.close()

    remaining = attempts - new_used
    print(f"user={user_id}: used={current_used} → {new_used} (남은: {remaining})")


def main():
    parser = argparse.ArgumentParser(description="Tool 테스트용 DB 유틸리티")
    parser.add_argument("--show", action="store_true", help="현재 DB 상태 조회")
    parser.add_argument("--init", action="store_true", help="더미 데이터 초기화")
    parser.add_argument("--use-review", nargs=2, metavar=("USER_ID", "COUNT"),
                        help="코드 리뷰 N회 사용 (남은 횟수 감소)")
    parser.add_argument("--use-practice", nargs=2, metavar=("USER_ID", "COUNT"),
                        help="사전연습 N회 사용 (남은 횟수 감소)")

    args = parser.parse_args()

    # 아무 옵션 없으면 init
    if not any([args.show, args.init, args.use_review, args.use_practice]):
        init_dummy()
        return

    if args.init:
        init_dummy()

    if args.use_review:
        uid, cnt = int(args.use_review[0]), int(args.use_review[1])
        use_review(uid, cnt)

    if args.use_practice:
        uid, cnt = int(args.use_practice[0]), int(args.use_practice[1])
        use_practice(uid, cnt)

    if args.show or args.use_review or args.use_practice:
        print()
        show_status()


if __name__ == "__main__":
    main()
