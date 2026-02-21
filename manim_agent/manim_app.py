import streamlit as st
from pathlib import Path
from python_run_agent import build_graph

st.set_page_config(page_title="Manim AI Agent", page_icon="🎬", layout="centered")

st.title("🎬 Manim AI Animation Generator")
st.caption("Generate Manim animations using AI with automatic debugging & retries")
st.divider()

user_query = st.text_area("🧠 Animation request", height=150,
                           placeholder="Explain Neural Networks using Manim...")

col1, col2 = st.columns(2)
scene_name  = col1.text_input("🎭 Scene Class Name", value="NeuralNetworkScene")
max_retries = col2.number_input("🔄 Max Retries", min_value=1, max_value=10, value=3)

generate = st.button("🚀 Generate Animation", use_container_width=True)

if generate:
    if not user_query.strip():
        st.warning("Please enter an animation request.")
        st.stop()

    state = {
        "user_query":       user_query,
        "generated_code":   "",
        "scene_name":       scene_name,
        "execution_output": {},
        "final_answer":     "",
        "video_path":       "",
        "retry_count":      0,
        "max_retries":      int(max_retries),
        "error_history":    [],
    }

    with st.spinner("🤖 Running agent…"):
        graph  = build_graph()
        result = graph.invoke(state)

    st.divider()

    # ── Metrics ──────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Retries Used",  result.get("retry_count", 0))
    m2.metric("Max Retries",   result.get("max_retries", max_retries))
    errors = result.get("error_history", [])
    m3.metric("Errors Caught", len(errors))

    # ── Final answer ─────────────────────────────────────────────────────────
    final = result.get("final_answer", "")
    if final:
        st.success(final)
    else:
        st.warning("No final answer returned.")

    # ── Generated code ───────────────────────────────────────────────────────
    code = result.get("generated_code", "")
    if code:
        with st.expander("💻 Generated Manim Code"):
            st.code(code, language="python")

    # ── Errors ───────────────────────────────────────────────────────────────
    if errors:
        with st.expander(f"⚠️ Error History ({len(errors)})"):
            for i, err in enumerate(errors, 1):
                st.markdown(f"**Error {i}**")
                st.code(str(err), language="text")

    # ── Video ─────────────────────────────────────────────────────────────────
    video_path = result.get("video_path", "")
    if video_path and Path(video_path).exists():
        st.subheader("🎥 Generated Video")
        st.video(video_path)
    else:
        st.error("❌ Video not generated.")
        if video_path:
            st.caption(f"Expected path: `{video_path}`")