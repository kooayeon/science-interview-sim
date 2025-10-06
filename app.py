import io
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
if not coach_comment: # 직접 코멘트가 없을때만 자동 피드백 우선
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
