import io
import os
import time
import base64
import json
from typing import List, Tuple

import streamlit as st
import pandas as pd
from gtts import gTTS
from pydub import AudioSegment
import pydub

# ffmpeg 위치 지정 (Cloud 환경 대비)
AudioSegment.converter = "/usr/bin/ffmpeg"
AudioSegment.ffprobe = "/usr/bin/ffprobe"

from streamlit_audiorecorder import audiorecorder

# OpenAI (SDK >= 1.30)
try:
    from openai import OpenAI
    _OPENAI_SDK_OK = True
except Exception:
    _OPENAI_SDK_OK = False

# =============================
# 0) 기본 설정
# =============================
st.set_page_config(page_title="과학고 면접 시뮬레이터", page_icon="🎤", layout="wide")

# Secrets에서 API 키 읽기
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
HAS_OPENAI = bool(OPENAI_API_KEY and _OPENAI_SDK_OK)

if HAS_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# 1) 질문 로딩
# =============================
DEFAULT_QUESTIONS = [
    {"question": "자기소개를 해보세요.", "category": "공통"},
    {"question": "최근 흥미롭게 본 과학 기사 하나를 설명하고, 왜 흥미로웠는지 말해보세요.", "category": "시사과학"},
    {"question": "고등학교에서 가장 자신있는 과목과 그 이유는?", "category": "학업"},
    {"question": "실패 경험을 하나 말하고 무엇을 배웠는지 설명해보세요.", "category": "태도"},
]


def parse_questions(file_bytes: bytes, filename: str) -> List[dict]:
    """CSV(.csv) 또는 TXT(.txt) 업로드를 파싱하여 [{question, category}] 리스트로 반환.
    - CSV: columns: question, category
    - TXT: "카테고리: 질문" 또는 "질문"(카테고리 미상은 "기타")
    """
    name = filename.lower()
    out = []
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        qcol = "question" if "question" in df.columns else df.columns[0]
        ccol = "category" if "category" in df.columns else None
        for _, row in df.iterrows():
            q = str(row[qcol]).strip()
            c = str(row[ccol]).strip() if ccol and pd.notna(row[ccol]) else "기타"
            if q:
                out.append({"question": q, "category": c})
    elif name.endswith(".txt"):
        text = io.BytesIO(file_bytes).read().decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                c, q = line.split(":", 1)
                out.append({"question": q.strip(), "category": c.strip() or "기타"})
            else:
                out.append({"question": line, "category": "기타"})
    return out


# =============================
# 2) 세션 상태 관리
# =============================
STATE_KEYS = [
    "questions", "order", "idx", "records",
    "timer_sec", "remaining", "timer_running",
    "auto_flow", "shuffle", "category_filter",
    "start_ts"
]


def init_state():
    if "__inited" in st.session_state:
        return

    st.session_state.questions = DEFAULT_QUESTIONS.copy()
    st.session_state.category_filter = "전체"
    st.session_state.shuffle = False
    st.session_state.auto_flow = True

    st.session_state.timer_sec = 90
    st.session_state.remaining = st.session_state.timer_sec
    st.session_state.timer_running = False
    st.session_state.start_ts = None

    st.session_state.records = []  # list of dicts
    rebuild_order()
    st.session_state.idx = 0

    st.session_state.__inited = True


def rebuild_order():
    qs = st.session_state.questions
    cf = st.session_state.get("category_filter", "전체")
    if cf and cf != "전체":
        filtered = [i for i, q in enumerate(qs) if q.get("category") == cf]
    else:
        filtered = list(range(len(qs)))
    if st.session_state.get("shuffle", False):
        import random
        random.shuffle(filtered)
    st.session_state.order = filtered


# =============================
# 3) 타이머 블록
# =============================

def timer_block():
    col1, col2, col3, col4 = st.columns([1,1,1,2])
    with col1:
        if st.button("⏱️ 시작", disabled=st.session_state.timer_running):
            st.session_state.timer_running = True
            st.session_state.start_ts = time.time()
    with col2:
        if st.button("⏸️ 정지", disabled=not st.session_state.timer_running):
            st.session_state.timer_running = False
            st.session_state.remaining = max(0, st.session_state.remaining - int(time.time() - st.session_state.start_ts))
    with col3:
        if st.button("🔄 리셋"):
            st.session_state.timer_running = False
            st.session_state.remaining = st.session_state.timer_sec
            st.session_state.start_ts = None

    # 진행 표시
    if st.session_state.timer_running and st.session_state.start_ts:
        elapsed = int(time.time() - st.session_state.start_ts)
        left = max(0, st.session_state.timer_sec - elapsed)
        st.session_state.remaining = left
        if left == 0:
            st.session_state.timer_running = False
    st.progress((st.session_state.timer_sec - st.session_state.remaining) / max(1, st.session_state.timer_sec))
    st.caption(f"남은 시간: {st.session_state.remaining}초 / 설정: {st.session_state.timer_sec}초")

    # 1초 주기 rerun
    if st.session_state.timer_running:
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()


# =============================
# 4) 음성/TTS/STT/GPT 유틸
# =============================

def tts_question(text: str):
    try:
        tts = gTTS(text=text, lang="ko")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS 오류: {e}")


def _audiosegment_to_wav_bytes(seg: AudioSegment) -> bytes:
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


def stt_whisper(wav_bytes: bytes) -> str:
    """OpenAI Whisper STT. 키 없으면 빈 문자열 반환."""
    if not HAS_OPENAI:
        return ""
    try:
        # SDK >=1.30 예시: audio.transcriptions.create
        file_tuple = ("audio.wav", wav_bytes, "audio/wav")
        res = client.audio.transcriptions.create(
            model="whisper-1",  # 필요 시 최신 모델명으로 교체
            file=file_tuple,
            language="ko"
        )
        # SDK별 반환 구조 차이를 흡수
        text = getattr(res, "text", None) or (res.get("text") if isinstance(res, dict) else None)
        return text or ""
    except Exception as e:
        st.warning(f"STT 호출 실패: {e}")
        return ""


def gpt_feedback(question: str, answer: str) -> dict:
    """GPT로 4항목 피드백 생성. 키 없으면 빈 dict"""
    if not HAS_OPENAI or not answer.strip():
        return {}
    sys = (
        "당신은 과학고 면접 코치입니다. 응답의 논리, 개념 정확성, 태도, 명료성을 짧고 구체적으로 평가하세요. "
        "각 항목당 1~2문장, 마지막에 총평 1문장. 점수는 5점 만점 정수."
    )
    usr = (
        f"[질문]\n{question}\n\n[답변]\n{answer}\n\n"
        "다음 형식의 JSON으로만 반환:\n"
        "{\"logic\":{\"score\":int,\"comment\":str},\"concept\":{...},\"attitude\":{...},\"clarity\":{...},\"overall\":str}"
    )
    try:
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            temperature=0.3,
        )
        content = chat.choices[0].message.content
        data = json.loads(content)
        return data
    except Exception as e:
        st.info(f"GPT 피드백 생성 실패: {e}")
        return {}


# =============================
# 5) 기록/리포트
# =============================

def save_record(question: str, answer: str, sec_used: int, rubric: dict, coach_comment: str, auto_fb: dict):
    st.session_state.records.append({
        "question": question,
        "answer": answer,
        "sec_used": sec_used,
        "rubric": rubric,
        "coach_comment": coach_comment,
        "auto_feedback": auto_fb,
    })


def to_csv_bytes() -> bytes:
    rows = []
    for r in st.session_state.records:
        rows.append({
            "question": r["question"],
            "answer": r["answer"],
            "sec_used": r["sec_used"],
            "logic": r.get("rubric",{}).get("logic", 0),
            "concept": r.get("rubric",{}).get("concept", 0),
            "attitude": r.get("rubric",{}).get("attitude", 0),
            "clarity": r.get("rubric",{}).get("clarity", 0),
            "coach_comment": r.get("coach_comment", ""),
            "auto_feedback": json.dumps(r.get("auto_feedback", {}), ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8-sig")


def to_markdown_bytes() -> bytes:
    lines = ["# 과학고 면접 연습 리포트\n"]
    for i, r in enumerate(st.session_state.records, 1):
        lines.append(f"## Q{i}. {r['question']}")
        lines.append(f"- 소요시간: {r['sec_used']}초")
        rb = r.get("rubric", {})
        lines.append(f"- 루브릭: 논리 {rb.get('logic',0)}/5, 개념 {rb.get('concept',0)}/5, 태도 {rb.get('attitude',0)}/5, 명료성 {rb.get('clarity',0)}/5")
        if r.get("coach_comment"):
            lines.append(f"- 코멘트: {r['coach_comment']}")
        if r.get("auto_feedback"):
            lines.append("<details><summary>자동 피드백</summary>")
            lines.append("\n```json\n" + json.dumps(r["auto_feedback"], ensure_ascii=False, indent=2) + "\n```\n")
            lines.append("</details>")
        lines.append("\n**Answer**\n\n" + (r['answer'] or "(패스)"))
        lines.append("\n---\n")
    md = "\n".join(lines)
    return md.encode("utf-8")


# =============================
# 6) 메인 UI
# =============================

def main():
    init_state()

    # ----- 사이드바 -----
    with st.sidebar:
        st.header("⚙️ 설정")
        up = st.file_uploader("질문지 업로드 (CSV/TXT)", type=["csv","txt"], accept_multiple_files=False)
        if up is not None:
            try:
                qs = parse_questions(up.read(), up.name)
                if qs:
                    st.session_state.questions = qs
                    # 카테고리 목록 재생성
                    st.session_state.category_filter = "전체"
                    st.session_state.shuffle = False
                    rebuild_order()
                    st.success(f"질문 {len(qs)}개 불러옴")
                else:
                    st.warning("질문을 찾지 못했습니다.")
            except Exception as e:
                st.error(f"업로드 파싱 실패: {e}")

        # 카테고리 필터
        cats = ["전체"] + sorted(list({q.get("category","기타") for q in st.session_state.questions}))
        sel = st.selectbox("카테고리", options=cats, index=cats.index(st.session_state.category_filter) if st.session_state.category_filter in cats else 0)
        if sel != st.session_state.category_filter:
            st.session_state.category_filter = sel
            rebuild_order()

        st.session_state.shuffle = st.toggle("셔플", value=st.session_state.shuffle, help="문항 순서를 랜덤화")
        if st.button("순서 재생성"):
            rebuild_order()

        st.session_state.auto_flow = st.toggle("제출/패스 시 자동 다음", value=st.session_state.auto_flow)

        st.session_state.timer_sec = st.slider("답변 시간(초)", 30, 300, st.session_state.timer_sec, step=15)
        if st.button("세션 리셋"):
            for k in STATE_KEYS:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

        st.divider()
        st.subheader("📥 리포트 다운로드")
        colA, colB = st.columns(2)
        with colA:
            st.download_button("CSV", to_csv_bytes, file_name="interview_report.csv", mime="text/csv")
        with colB:
            st.download_button("Markdown", to_markdown_bytes, file_name="interview_report.md", mime="text/markdown")

        st.divider()
        st.caption("🔑 OpenAI 키 상태:")
        if HAS_OPENAI:
            st.success("감지됨 (STT/피드백 활성)")
        else:
            if not _OPENAI_SDK_OK:
                st.warning("openai SDK 로딩 실패. requirements 버전 확인")
            st.error("미설정 (텍스트 기능만 사용 가능)")

    # ----- 본문 -----
    st.title("🎤 과학고 면접 시뮬레이터")

    if not st.session_state.order:
        st.info("선택된 카테고리에 해당하는 질문이 없습니다. 필터를 조정하세요.")
        return

    # 현재 문항
    idx = st.session_state.idx
    if idx >= len(st.session_state.order):
        st.success("모든 문항을 완료했습니다! 사이드바에서 리포트를 다운로드하세요.")
        return

    q = st.session_state.questions[ st.session_state.order[idx] ]
    st.subheader(f"Q{idx+1}. {q['question']}")

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🔊 질문 듣기"):
            tts_question(q["question"])    
    with c2:
        if st.button("⏭️ 다음으로 건너뛰기"):
            st.session_state.idx = min(idx+1, len(st.session_state.order))
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    with st.expander("답변 템플릿 가이드"):
        st.markdown("""
        - **서론(5~10초)**: 질문 재확인 + 핵심 결론 1문장
        - **본론(60~70초)**: 근거 2~3개, 예시 1개
        - **결론(10~15초)**: 요약 + 확장/한계 언급
        """)

    timer_block()

    st.markdown("### 🎙️ 음성 녹음 / STT")
    with st.expander("마이크로 녹음하기"):
        audio = audiorecorder("Start", "Stop")
        st.caption("Stop 후 아래에서 파형 미리보기 > 텍스트 변환")
        if len(audio) > 0:
            st.audio(audio.export().read(), format="audio/wav")  # 파형 재생
            if st.button("🪄 음성을 텍스트로 변환(STT)"):
                try:
                    wav_bytes = _audiosegment_to_wav_bytes(audio)
                    text = stt_whisper(wav_bytes)
                    if text:
                        st.session_state.answer_text = text
                        st.success("변환 완료. 아래 답변 입력란에 채워졌습니다.")
                    else:
                        st.warning("변환 결과가 비어있습니다. (키/네트워크/음질 확인)")
                except Exception as e:
                    st.error(f"STT 변환 실패: {e}")

    st.markdown("### ✍️ 텍스트 답변")
    default_ans = st.session_state.get("answer_text", "")
    answer = st.text_area("여기에 답변을 입력하거나, STT로 채워진 텍스트를 수정하세요.", value=default_ans, height=160)

    st.markdown("### 🧭 루브릭 평가 (자기/코치)")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        logic = st.slider("논리", 0, 5, 3)
    with r2:
        concept = st.slider("개념", 0, 5, 3)
    with r3:
        attitude = st.slider("태도", 0, 5, 3)
    with r4:
        clarity = st.slider("명료성", 0, 5, 3)

    coach_comment = st.text_input("직접 코멘트(선택)")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("✅ 제출"):
            # 소요시간 계산
            used = st.session_state.timer_sec - st.session_state.remaining
            used = max(0, used)

            # 자동 피드백
            auto_fb = {}
            if not coach_comment:  # 직접 코멘트가 없을때만 자동 피드백 우선
                auto_fb = gpt_feedback(q["question"], answer)

            save_record(
                question=q["question"],
                answer=answer,
                sec_used=used,
                rubric={"logic":logic, "concept":concept, "attitude":attitude, "clarity":clarity},
                coach_comment=coach_comment,
                auto_fb=auto_fb,
            )

            # 타이머 리셋
            st.session_state.timer_running = False
            st.session_state.remaining = st.session_state.timer_sec
            st.session_state.start_ts = None

            # 자동 다음
            if st.session_state.auto_flow:
                st.session_state.idx = min(idx+1, len(st.session_state.order))
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    with colB:
        if st.button("🧼 타이머 리셋"):
            st.session_state.timer_running = False
            st.session_state.remaining = st.session_state.timer_sec
            st.session_state.start_ts = None

    with colC:
        if st.button("🚫 패스"):
            save_record(q["question"], "", 0, {"logic":0,"concept":0,"attitude":0,"clarity":0}, "(패스)", {})
            if st.session_state.auto_flow:
                st.session_state.idx = min(idx+1, len(st.session_state.order))
            st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()

    with st.expander("📊 진행 현황"):
        if st.session_state.records:
            view = []
            for i, r in enumerate(st.session_state.records, 1):
                view.append({
                    "#": i,
                    "질문": r["question"][:30] + ("…" if len(r["question"])>30 else ""),
                    "시간(초)": r["sec_used"],
                    "논리": r["rubric"].get("logic",0),
                    "개념": r["rubric"].get("concept",0),
                    "태도": r["rubric"].get("attitude",0),
                    "명료성": r["rubric"].get("clarity",0),
                    "코멘트": (r.get("coach_comment") or "")[:20]
                })
            st.dataframe(pd.DataFrame(view))
        else:
            st.caption("아직 기록이 없습니다.")


if __name__ == "__main__":
    main()
