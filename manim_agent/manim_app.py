import streamlit as st
from pathlib import Path
import time

# import your graph builder file
from python_run_agent import build_graph   # <-- change filename if needed

st.set_page_config(page_title="Manim AI Agent", layout="wide")

st.title("🎬 Manim AI Animation Generator")

# User input
user_query = st.text_area(
    "Enter your animation request:",
    height=150,
    placeholder="Explain Neural Networks using manim..."
)

scene_name = st.text_input(
    "Scene Class Name",
    value="NeuralNetworkScene"
)

generate = st.button("Generate Animation")

if generate and user_query:

    with st.spinner("🤖 Generating animation... This may take 1-3 minutes..."):
        
        graph = build_graph()

        state = {
            "user_query": user_query,
            "generated_code": "",
            "scene_name": scene_name,
            "execution_output": {},
            "final_answer": "",
            "video_path": "",
            "retry_count": 0,
            "max_retries": 3,
            "error_history": []
        }

        response = graph.invoke(state)

    st.subheader("📜 Final Status")
    st.code(response["final_answer"])

    # Display video if success
    video_path = response.get("video_path")

    if video_path and Path(video_path).exists():
        st.subheader("🎥 Generated Video")
        st.video(video_path)
    else:
        st.error("Video not generated.")
