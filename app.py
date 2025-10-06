# science_high_interview_sim.py
# ------------------------------------------------------------
# 과학고 면접 연습용 Streamlit 앱 (단일 파일)
# - 질문 은행 불러오기(.txt/.csv) 또는 기본 질문 사용
# - 1문항씩 제시, 답변 작성 + 60초 카운트다운(조절 가능)
# - 루브릭(논리, 과학개념, 태도, 명료성)으로 자기/코치 평가
# - 기록 자동 저장, 세션 종료 시 CSV/Markdown 리포트 다운로드
# - 카테고리 필터, 무작위 출제, 오토플로우(제출 시 자동 다음)
# ------------------------------------------------------------

import io
import random
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# -----------------------------
# 기본 질문 세트 (필요시 수정)
# -----------------------------
DEFAULT_QUESTIONS = [
    {"category": "인성", "question": "과학고에 지원한 동기는 무엇인가요?"},
    {"category": "인성", "question": "팀 프로젝트에서 갈등이 있었을 때 어떻게 해결했나요?"},
    {"category": "탐구", "question": "최근에 수행한 탐구/실험 주제와 가설, 변인 통제를 설명해 보세요."},
    {"category": "탐구", "question": "예상과 다른 결과가 나왔던 경험과 원인 분석, 개선안을 말해 보세요."},
    {"category": "과학", "question": "빛의 굴절 현상을 일상 사례로 설명해 보세요."},
    {"category": "과학", "question": "기체의 압력과 부피 관계(보일 법칙)를 설명하고 실생활 적용 예를 드세요."},
    {"category": "수학", "question": "함수의 기울기의 의미를 그래프와 함께 말로 설명해 보세요."},
    {"category": "수학", "question": "수열에서 규칙성을 발견하는 본인만의 접근 과정을 설명해 보세요."},
]

# -----------------------------
# 유틸: 파일 파싱(.txt/.csv)
# -----------------------------

def parse_questions(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        # 기대 컬럼: question (필수), category (선택)
        if "question" not in df.columns:
            raise ValueError("CSV에 'question' 컬럼이 필요합니다.")
        if "category" not in df.columns:
            df["category"] = "일반"
        records = df[["category", "question"]].fillna("").to_dict(orient="records")
        return records
    elif name.endswith(".txt"):
        text = io.BytesIO(file_bytes).read().decode("utf-8")
        # 포맷1) '카테고리: 질문'  / 포맷2) '질문' (카테고리 없음)
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                cat, q = line.split(":", 1)
                records.append({"category": cat.strip() or "일반", "question": q.strip()})
            else:
                records.append({"category": "일반", "question": line})
        if not records:
            raise ValueError("TXT에서 유효한 질문을 찾지 못했습니다.")
        return records
    else:
        raise ValueError("지원되는 파일 형식은 .txt, .csv 입니다.")

# -----------------------------
# 세션 스토리지 초기화
# -----------------------------

def init_state():
    st.session_state.setdefault("questions", DEFAULT_QUESTIONS.copy())
    st.session_state.setdefault("order", list(range(len(st.session_state["questions"]))))
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("records", [])  # 답변/평가 로그
    st.session_state.setdefault("timer_sec", 60)
    st.session_state.setdefault("auto_flow", True)
    st.session_state.setdefault("shuffle", False)
    st.session_state.setdefault("category_filter", "전체")
    st.session_state.setdefault("started_at", None)

# -----------------------------
# 타이머 컴포넌트
# -----------------------------

def countdown(key: str = "remaining"):
    # 남은 초를 session_state에 저장하고, 1초마다 재렌더
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get("timer_sec", 60)
    remaining = st.session_state[key]

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.progress(max(0, min(100, int((remaining / st.session_state["timer_sec"]) * 100))))
    with col_t2:
        st.metric("남은 시간(s)", remaining)

    # 자동 리프레시
    if remaining > 0:
        st.experimental_rerun  # no-op reference for clarity
        st.autorefresh = st.experimental_get_query_params  # to satisfy linters
        st.experimental_set_query_params(_=datetime.now().strftime("%H%M%S"))
        st.session_state[key] = remaining - 1

# -----------------------------
# 데이터프레임 -> CSV/MD 바이트
# -----------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_markdown_bytes(df: pd.DataFrame) -> bytes:
    md = ["# 과학고 면접 연습 리포트\n"]
    for i, row in df.iterrows():
        md.append(f"## Q{i+1}. {row['question']}")
        md.append(f"- 카테고리: {row['category']}")
        md.append(f"- 소요시간: {row['duration_sec']}초")
        md.append(
            f"- 점수(1~5): 논리 {row['score_logic']}, 개념 {row['score_concept']}, 태도 {row['score_attitude']}, 명료성 {row['score_clarity']}")
        md.append(f"- 총평: {row['coach_comment']}")
        md.append("\n**답변:**\n")
        md.append(row["answer"] or "(미작성)")
        md.append("\n---\n")
    return "\n".join(md).encode("utf-8")

# -----------------------------
# 메인 앱
# -----------------------------

def main():
    st.set_page_config(page_title="과학고 면접 시뮬레이터", page_icon="🧪", layout="wide")
    init_state()

    with st.sidebar:
        st.title("🧪 과학고 면접 시뮬레이터")
        st.caption("텍스트 기반 1문항 진행형 / 평가 및 리포트 지원")

        up = st.file_uploader("질문지 업로드 (.txt 또는 .csv)", type=["txt", "csv"])
        if up:
            try:
                st.session_state["questions"] = parse_questions(up.read(), up.name)
                # 카테고리 필터 초기화
                st.session_state["category_filter"] = "전체"
                st.session_state["order"] = list(range(len(st.session_state["questions"])) )
                st.session_state["idx"] = 0
                st.success(f"질문 {len(st.session_state['questions'])}개 로드 완료")
            except Exception as e:
                st.error(f"업로드 실패: {e}")

        # 카테고리 필터
        cats = ["전체"] + sorted({q["category"] for q in st.session_state["questions"]})
        st.session_state["category_filter"] = st.selectbox("카테고리 필터", cats, index=0)

        # 타이머 & 설정
        st.session_state["timer_sec"] = st.slider("답변 시간(초)", 15, 180, st.session_state["timer_sec"])  # 15~180초
        st.session_state["auto_flow"] = st.toggle("제출 시 자동 다음으로", value=st.session_state["auto_flow"])
        st.session_state["shuffle"] = st.toggle("무작위 출제", value=st.session_state["shuffle"])

        if st.button("새 세션 시작/리셋", type="primary"):
            # 필터 반영된 순서 만들기
            indices = [i for i, q in enumerate(st.session_state["questions"]) if
                       st.session_state["category_filter"] in ("전체", q["category"])]
            if st.session_state["shuffle"]:
                random.shuffle(indices)
            st.session_state["order"] = indices
            st.session_state["idx"] = 0
            st.session_state["records"] = []
            st.session_state["started_at"] = datetime.now().isoformat()
            st.session_state["remaining"] = st.session_state["timer_sec"]
            st.success("세션이 초기화되었습니다. 화이팅!")

        # 내보내기
        if st.session_state.get("records"):
            df = pd.DataFrame(st.session_state["records"])  # type: ignore
            st.download_button("CSV 다운로드", data=to_csv_bytes(df), file_name="interview_report.csv")
            st.download_button("Markdown 다운로드", data=to_markdown_bytes(df), file_name="interview_report.md")

        st.markdown("---")
        st.caption("Tip: CSV 업로드 시 컬럼명은 question(필수), category(선택)")

    # 본문 - 현재 문항 표시
    order = st.session_state.get("order", [])
    if not order:
        st.info("좌측에서 '새 세션 시작/리셋'을 눌러 시작하세요.")
        return

    cur_pos = st.session_state.get("idx", 0)
    if cur_pos >= len(order):
        st.success("🎉 모든 문항을 완료했습니다!")
        if st.session_state.get("records"):
            df = pd.DataFrame(st.session_state["records"])  # type: ignore
            st.dataframe(df)
        return

    q_idx = order[cur_pos]
    q = st.session_state["questions"][q_idx]

    header_left, header_right = st.columns([6, 1])
    with header_left:
        st.subheader(f"Q{cur_pos + 1} / {len(order)}  ·  [{q['category']}]  {q['question']}")
    with header_right:
        st.button("다음으로 건너뛰기", on_click=lambda: st.session_state.update({"idx": cur_pos + 1, "remaining": st.session_state["timer_sec"]}))

    # 힌트/구조 템플릿
    with st.expander("답변 구조 템플릿 보기"):
        st.markdown(
            """
            **권장 구조(질문 유형 공통):**
            1) 배경·관심 계기 → 2) 핵심 개념/가설 → 3) 근거·과정(실험/추론) → 4) 결과·한계 → 5) 다음 개선·확장
            
            **예시 문장틀:**
            - "제가 이렇게 생각한 이유는 … 이고, 이를 검증하기 위해 … 방법을 사용했습니다."
            - "예상과 다르게 나온 부분은 … 때문이라고 판단했고, 다음엔 …로 개선하겠습니다."
            """
        )

    # 카운트다운
    st.info("타이머가 0이 되어도 답변 작성은 가능합니다. 긴장감 조절용이에요.")
    if "remaining" not in st.session_state:
        st.session_state["remaining"] = st.session_state["timer_sec"]
    countdown("remaining")

    # 답변 입력
    answer = st.text_area("답변 입력", key=f"ans_{q_idx}", height=180, placeholder="구조를 따라 차분히 서술해 보세요…")

    # 평가 루브릭
    with st.expander("자기/코치 평가 (선택)"):
        col1, col2, col3, col4 = st.columns(4)
        score_logic = col1.slider("논리", 1, 5, 3)
        score_concept = col2.slider("과학개념", 1, 5, 3)
        score_attitude = col3.slider("태도", 1, 5, 3)
        score_clarity = col4.slider("명료성", 1, 5, 3)
        coach_comment = st.text_area("총평/피드백", height=100, placeholder="핵심 강점과 다음에 보완할 1가지를 적어주세요.")

    # 제출/컨트롤
    btn_cols = st.columns([1, 1, 1])
    submit = btn_cols[0].button("제출 및 저장", type="primary")
    reset_timer = btn_cols[1].button("타이머 리셋")
    pass_q = btn_cols[2].button("패스(미답변)")

    if reset_timer:
        st.session_state["remaining"] = st.session_state["timer_sec"]
        st.toast("타이머를 리셋했습니다.")

    def save_record(missed: bool = False):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": q["category"],
            "question": q["question"],
            "answer": "" if missed else answer.strip(),
            "duration_sec": int(st.session_state["timer_sec"] - st.session_state.get("remaining", 0)),
            "score_logic": score_logic,
            "score_concept": score_concept,
            "score_attitude": score_attitude,
            "score_clarity": score_clarity,
            "coach_comment": coach_comment.strip(),
        }
        st.session_state["records"].append(record)

    if submit:
        save_record(missed=False)
        st.success("저장 완료!")
        # 다음 문항 이동
        st.session_state["idx"] = cur_pos + 1
        st.session_state["remaining"] = st.session_state["timer_sec"]
        if st.session_state["auto_flow"]:
            st.experimental_rerun()

    if pass_q:
        save_record(missed=True)
        st.warning("패스로 기록했습니다.")
        st.session_state["idx"] = cur_pos + 1
        st.session_state["remaining"] = st.session_state["timer_sec"]
        if st.session_state["auto_flow"]:
            st.experimental_rerun()

    # 하단 - 진행 현황
    with st.expander("진행 현황 / 기록 보기"):
        if st.session_state.get("records"):
            df = pd.DataFrame(st.session_state["records"])  # type: ignore
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("아직 저장된 기록이 없습니다.")


if __name__ == "__main__":
    main()
