# app.py — 과학고 면접 시뮬레이터 (Streamlit 단일파일)
# ------------------------------------------------------------
# 웹 배포용(Community Cloud) 안정화 + 음성/STT/TTS/GPT 피드백 통합 + 자동 TTS/퀵 STT
# - 질문 화면 진입 시 자동 TTS 재생(autoplay)
# - "🎙️ 원클릭 녹음→자동 STT" 버튼: 녹음 후 자동으로 Whisper 변환
# - experimental_* API 제거, st.rerun() 사용
# - TXT/CSV 질문 업로드, 카테고리 필터/셔플, 자동 다음 이동
# - 타이머: 시작/정지/리셋(1초 주기 안전 rerun)
# ------------------------------------------------------------

import io
import base64
import random
import time
from datetime import datetime
from typing import List, Dict

import pandas as pd
import streamlit as st
from io import BytesIO

# 음성/AI 관련(선택적 의존성)
try:
    from openai import OpenAI
except Exception:  # 패키지 미설치 대비
    OpenAI = None  # type: ignore

try:
    from gtts import gTTS
except Exception:  # 패키지 미설치 대비
    gTTS = None  # type: ignore

try:
    from audiorecorder import audiorecorder
except Exception:
    audiorecorder = None  # type: ignore

# ------------------------------------------------------------
# OpenAI 클라이언트 (Secrets에 OPENAI_API_KEY 저장 필요)
# ------------------------------------------------------------
OPENAI_API_KEY = None
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
except Exception:
    OPENAI_API_KEY = None

client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# -----------------------------
# 기본 질문 세트
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
# 파일 파싱 (.txt / .csv)
# -----------------------------

def parse_questions(file_bytes: bytes, filename: str) -> List[Dict[str, str]]:
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        if "question" not in df.columns:
            raise ValueError("CSV에 'question' 컬럼이 필요합니다.")
        if "category" not in df.columns:
            df["category"] = "일반"
        return df[["category", "question"]].fillna("").to_dict(orient="records")
    elif name.endswith(".txt"):
        text = io.BytesIO(file_bytes).read().decode("utf-8")
        records: List[Dict[str, str]] = []
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
# 세션 스테이트 초기화
# -----------------------------

def init_state():
    st.session_state.setdefault("questions", DEFAULT_QUESTIONS.copy())
    st.session_state.setdefault("order", list(range(len(st.session_state["questions"]))))
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("timer_sec", 60)
    st.session_state.setdefault("remaining", 60)
    st.session_state.setdefault("timer_running", False)
    st.session_state.setdefault("auto_flow", True)
    st.session_state.setdefault("shuffle", False)
    st.session_state.setdefault("category_filter", "전체")
    st.session_state.setdefault("started_at", None)
    st.session_state.setdefault("last_tts_qidx", -1)  # 자동 TTS 재생 제어
    st.session_state.setdefault("quick_rec", False)   # 원클릭 녹음 모드

# -----------------------------
# 타이머 표시 & 틱
# -----------------------------

def timer_block():
    total = max(1, int(st.session_state.get("timer_sec", 60)))
    remaining = int(st.session_state.get("remaining", total))

    prog = int((remaining / total) * 100)
    st.progress(max(0, min(100, prog)))
    st.metric("남은 시간(s)", remaining)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("타이머 시작", use_container_width=True):
            st.session_state["timer_running"] = True
            st.rerun()
    with c2:
        if st.button("타이머 정지", use_container_width=True):
            st.session_state["timer_running"] = False
            st.rerun()
    with c3:
        if st.button("타이머 리셋", use_container_width=True):
            st.session_state["remaining"] = total
            st.session_state["timer_running"] = False
            st.toast("타이머를 리셋했습니다.")
            st.rerun()

    # 1초 틱: 실행 중이면 sleep 후 rerun
    if st.session_state.get("timer_running", False):
        if remaining > 0:
            time.sleep(1)
            st.session_state["remaining"] = remaining - 1
            st.rerun()
        else:
            st.session_state["timer_running"] = False

# -----------------------------
# 리포트 내보내기
# -----------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_markdown_bytes(df: pd.DataFrame) -> bytes:
    md_lines = ["# 과학고 면접 연습 리포트\n"]
    for i, row in df.iterrows():
        md_lines.append(f"## Q{i+1}. {row['question']}")
        md_lines.append(f"- 카테고리: {row['category']}")
        md_lines.append(f"- 소요시간: {row['duration_sec']}초")
        md_lines.append(
            f"- 점수(1~5): 논리 {row['score_logic']}, 개념 {row['score_concept']}, 태도 {row['score_attitude']}, 명료성 {row['score_clarity']}"
        )
        md_lines.append(f"- 총평: {row['coach_comment']}")
        md_lines.append("\n**답변:**\n")
        md_lines.append(row.get("answer", "") or "(미작성)")
        md_lines.append("\n---\n")
    return "\n".join(md_lines).encode("utf-8")
def to_markdown_bytes(df: pd.DataFrame) -> bytes:
    md_lines = ["# 과학고 면접 연습 리포트\n"]
    for i, row in df.iterrows():
        md_lines.append(f"## Q{i+1}. {row['question']}")
        md_lines.append(f"- 카테고리: {row['category']}")
        md_lines.append(f"- 소요시간: {row['duration_sec']}초")
        md_lines.append(
            f"- 점수(1~5): 논리 {row['score_logic']}, 개념 {row['score_concept']}, 태도 {row['score_attitude']}, 명료성 {row['score_clarity']}"
        )
        md_lines.append(f"- 총평: {row['coach_comment']}")
        md_lines.append("\n**답변:**\n")
        md_lines.append(row.get("answer", "") or "(미작성)")
        md_lines.append("\n---\n")
    return "\n".join(md_lines).encode("utf-8")


# -----------------------------
# 음성/AI 유틸
# -----------------------------

def autoplay_audio(audio_bytes: bytes, mime: str = "audio/mp3", hidden: bool = True):
    """브라우저에서 자동 재생되는 <audio autoplay> 삽입"""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    style = "display:none;" if hidden else ""
    html = f"""
    <audio autoplay {'' if not hidden else ''} style='{style}'>
      <source src="data:{mime};base64,{b64}">
    </audio>
    """
    st.markdown(html, unsafe_allow_html=True)


def tts_question(text: str) -> bytes:
    """질문 텍스트를 mp3 바이트로 변환(gTTS)."""
    if not text or gTTS is None:
        return b""
    try:
        mp3_bytes = BytesIO()
        gTTS(text=text, lang="ko").write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)
        return mp3_bytes.read()
    except Exception:
        return b""


def stt_whisper(wav_bytes: bytes) -> str:
    """녹음된 음성(wav) -> Whisper API 자막 텍스트"""
    if not wav_bytes or client is None:
        return ""
    try:
        bio = BytesIO(wav_bytes)
        bio.name = "answer.wav"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=bio,
            language="ko",
        )
        return getattr(transcript, "text", "").strip()
    except Exception:
        return ""


def gpt_feedback(question: str, answer: str) -> str:
    """답변 자동 피드백(논리/개념/태도/명료성). 클라이언트 없으면 빈 문자열."""
    if client is None or not answer.strip():
        return ""

    sys_prompt = (
        "너는 과학고 면접관이다. 답변을 4가지 항목(논리, 과학개념, 태도, 명료성)으로 "
        "각 1~5점과 한 줄 코칭으로 간단히 평가하라. 총 평점도 1줄로."
    )

    user_prompt = "[질문]
{}

[답변]
{}

형식: 
- 논리: ?/5
- 과학개념: ?/5
- 태도: ?/5
- 명료성: ?/5
- 코칭 한 줄: ...
- 총평: ...".format(
        question, answer
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

# -----------------------------
# 메인 앱
# -----------------------------

def main():
    st.set_page_config(page_title="과학고 면접 시뮬레이터", page_icon="🧪", layout="wide")
    init_state()

    # 사이드바
    with st.sidebar:
        st.title("🧪 과학고 면접 시뮬레이터")
        st.caption("텍스트/음성 기반 1문항 진행형 · 평가 및 리포트")

        # API 키 상태 안내
        if client is None:
            st.warning("OpenAI API 키가 없어 STT/자동피드백은 비활성화됩니다. (Settings → Secrets에 OPENAI_API_KEY 등록)")
        else:
            st.success("OpenAI API 키 인식됨: STT/자동피드백 사용 가능!")

        up = st.file_uploader("질문지 업로드 (.txt 또는 .csv)", type=["txt", "csv"])
        if up:
            try:
                st.session_state["questions"] = parse_questions(up.read(), up.name)
                st.session_state["category_filter"] = "전체"
                st.session_state["order"] = list(range(len(st.session_state["questions"])))
                st.session_state["idx"] = 0
                st.session_state["records"] = []
                st.session_state["remaining"] = st.session_state["timer_sec"]
                st.success(f"질문 {len(st.session_state['questions'])}개 로드 완료")
            except Exception as e:
                st.error(f"업로드 실패: {e}")

        cats = ["전체"] + sorted({q["category"] for q in st.session_state["questions"]})
        st.session_state["category_filter"] = st.selectbox("카테고리 필터", cats, index=0)

        st.session_state["timer_sec"] = st.slider("답변 시간(초)", 15, 180, st.session_state["timer_sec"])  # 15~180초
        if not st.session_state.get("timer_running", False):
            st.session_state["remaining"] = st.session_state["timer_sec"]

        st.session_state["auto_flow"] = st.toggle("제출 시 자동 다음으로", value=st.session_state["auto_flow"])
        st.session_state["shuffle"] = st.toggle("무작위 출제", value=st.session_state["shuffle"])

        if st.button("새 세션 시작/리셋", type="primary"):
            indices = [i for i, q in enumerate(st.session_state["questions"]) if
                       st.session_state["category_filter"] in ("전체", q["category"])]
            if st.session_state["shuffle"]:
                random.shuffle(indices)
            st.session_state["order"] = indices
            st.session_state["idx"] = 0
            st.session_state["records"] = []
            st.session_state["started_at"] = datetime.now().isoformat()
            st.session_state["remaining"] = st.session_state["timer_sec"]
            st.session_state["timer_running"] = False
            st.session_state["last_tts_qidx"] = -1
            st.success("세션이 초기화되었습니다. 화이팅!")

        if st.session_state.get("records"):
            df = pd.DataFrame(st.session_state["records"])  # type: ignore
            st.download_button("CSV 다운로드", data=to_csv_bytes(df), file_name="interview_report.csv")
            st.download_button("Markdown 다운로드", data=to_markdown_bytes(df), file_name="interview_report.md")

        st.markdown("---")
        st.caption("Tip: CSV 업로드 시 컬럼명은 question(필수), category(선택)")

    # 본문
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

    header_left, header_right = st.columns([6, 2])
    with header_left:
        st.subheader(f"Q{cur_pos + 1} / {len(order)}  ·  [{q['category']}]  {q['question']}")
    with header_right:
        # 질문 듣기 (TTS) + 자동 재생 제어
        play_now = False
        if st.session_state.get("last_tts_qidx", -1) != q_idx:
            play_now = True
            st.session_state["last_tts_qidx"] = q_idx
        col_tts1, col_tts2 = st.columns(2)
        with col_tts1:
            if st.button("🔊 질문 듣기", use_container_width=True):
                play_now = True
        with col_tts2:
            if audiorecorder is not None and st.button("🎙️ 원클릭 녹음→자동 STT", use_container_width=True):
                st.session_state["quick_rec"] = True
                st.rerun()

        if play_now:
            mp3 = tts_question(q["question"])
            if mp3:
                autoplay_audio(mp3, mime="audio/mp3", hidden=True)

        st.button(
            "다음으로 건너뛰기",
            on_click=lambda: st.session_state.update({
                "idx": cur_pos + 1,
                "remaining": st.session_state["timer_sec"],
                "timer_running": False,
            }),
            use_container_width=True,
        )

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

    st.info("타이머가 0이 되어도 답변 작성은 가능합니다. 긴장감 조절용이에요.")
    timer_block()

    # 빠른 녹음 모드(버튼 한 번 → Stop 누르면 자동 STT)
    if st.session_state.get("quick_rec", False) and audiorecorder is not None:
        with st.container(border=True):
            st.markdown("**🎙️ 빠른 녹음 모드** — Stop을 누르면 자동으로 STT가 실행돼요.")
            quick_audio = audiorecorder("Start recording", "Stop recording")
            if quick_audio is not None and len(quick_audio) > 0:
                wav_io = BytesIO()
                quick_audio.export(wav_io, format="wav")
                wav_bytes = wav_io.getvalue()
                st.audio(wav_bytes, format="audio/wav")
                if client is None:
                    st.warning("OpenAI API 키가 없어 STT를 실행할 수 없습니다.")
                else:
                    text = stt_whisper(wav_bytes)
                    if text:
                        st.session_state[f"ans_{q_idx}"] = text
                        st.success("자막 변환 완료! 아래 답변 칸에 채워졌어요.")
                    else:
                        st.warning("자막 변환에 실패했어요. 다시 시도해 주세요.")
                # 한 번 실행 후 종료
                st.session_state["quick_rec"] = False
                st.rerun()

    # 음성 녹음 → STT 변환 (일반 모드)
    if audiorecorder is not None:
        with st.expander("🎙️ 음성으로 답변하기 / 자동 자막(STT)"):
            st.caption("Start → 말하기 → Stop 을 누르면 자막이 자동 채워집니다.")
            audio = audiorecorder("Start recording", "Stop recording")

            if audio is not None and len(audio) > 0:
                wav_bytes_io = BytesIO()
                audio.export(wav_bytes_io, format="wav")
                wav_bytes = wav_bytes_io.getvalue()

                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    st.audio(wav_bytes, format="audio/wav")
                with col_a2:
                    if st.button("자막 변환(STT)", use_container_width=True, disabled=(client is None)):
                        if client is None:
                            st.warning("OpenAI API 키가 없어 STT를 실행할 수 없습니다.")
                        else:
                            text = stt_whisper(wav_bytes)
                            if text:
                                st.session_state[f"ans_{q_idx}"] = text
                                st.success("자막 변환 완료! 아래 답변 창에 채워졌어요.")
                            else:
                                st.warning("자막 변환에 실패했어요. 다시 시도해 주세요.")
    else:
        st.caption("(옵션) 음성 녹음을 쓰려면 requirements.txt에 streamlit-audiorecorder, pydub을 추가하세요.")

    # 텍스트 답변 입력
    answer = st.text_area("답변 입력", key=f"ans_{q_idx}", height=180, placeholder="구조를 따라 차분히 서술해 보세요…")

    # 평가 루브릭
    with st.expander("자기/코치 평가 (선택)"):
        col1, col2, col3, col4 = st.columns(4)
        score_logic = col1.slider("논리", 1, 5, 3)
        score_concept = col2.slider("과학개념", 1, 5, 3)
        score_attitude = col3.slider("태도", 1, 5, 3)
        score_clarity = col4.slider("명료성", 1, 5, 3)
        coach_comment = st.text_area("총평/피드백", height=100, placeholder="핵심 강점과 다음에 보완할 1가지를 적어주세요.")

    btn_cols = st.columns([1, 1, 1])
    submit = btn_cols[0].button("제출 및 저장", type="primary")
    reset_timer_btn = btn_cols[1].button("타이머 리셋")
    pass_q = btn_cols[2].button("패스(미답변)")

    if reset_timer_btn:
        st.session_state["remaining"] = st.session_state["timer_sec"]
        st.session_state["timer_running"] = False
        st.toast("타이머를 리셋했습니다.")

    def save_record(missed: bool = False):
        # 자동 코멘트 (키 없으면 공백)
        auto_comment = gpt_feedback(q["question"], (answer or "").strip())
        final_comment = (coach_comment.strip() if coach_comment and coach_comment.strip() else auto_comment)

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": q["category"],
            "question": q["question"],
            "answer": "" if missed else ((answer or "").strip()),
            "duration_sec": int(st.session_state["timer_sec"] - st.session_state.get("remaining", 0)),
            "score_logic": score_logic,
            "score_concept": score_concept,
            "score_attitude": score_attitude,
            "score_clarity": score_clarity,
            "coach_comment": final_comment,
        }
        st.session_state["records"].append(record)

    if submit:
        save_record(missed=False)
        st.success("저장 완료!")
        st.session_state["idx"] = cur_pos + 1
        st.session_state["remaining"] = st.session_state["timer_sec"]
        st.session_state["timer_running"] = False
        st.session_state["quick_rec"] = False
        if st.session_state["auto_flow"]:
            st.rerun()

    if pass_q:
        save_record(missed=True)
        st.warning("패스로 기록했습니다.")
        st.session_state["idx"] = cur_pos + 1
        st.session_state["remaining"] = st.session_state["timer_sec"]
        st.session_state["timer_running"] = False
        st.session_state["quick_rec"] = False
        if st.session_state["auto_flow"]:
            st.rerun()

    with st.expander("진행 현황 / 기록 보기"):
        if st.session_state.get("records"):
            df = pd.DataFrame(st.session_state["records"])  # type: ignore
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("아직 저장된 기록이 없습니다.")


if __name__ == "__main__":
    main()
