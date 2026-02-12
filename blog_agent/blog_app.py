import streamlit as st
from blog_backend import build_graph

@st.cache_resource
def load_graph():
    return build_graph()

graph = load_graph()

st.set_page_config(page_title="AI Blog Generator", layout="wide")

st.title("📝 AI Blog Generator (LangGraph)")

blog_title = st.text_input(
    "Enter your blog title",
    placeholder="Building an END to END CNN workflow using PyTorch"
)

generate_btn = st.button("Generate Blog 🚀")

if generate_btn:

    if not blog_title.strip():
        st.warning("Please enter a blog title.")
    else:
        initial_state = {
            "topic": blog_title
        }

        st.subheader("🔄 Agent internal execution")

        log_box = st.empty()     # live log area
        logs = []

        final_state = None

        # ---- STREAM and capture final state ----
        for event in graph.stream(initial_state):
            # event is a dict like:
            # { "router": {...state...} }
            node_name = list(event.keys())[0]
            
            logs.append(f"▶ Running node: {node_name}")
            log_box.markdown("\n".join(logs))
            
            # Capture the state from the last event
            final_state = event[node_name]

        st.success("Blog generated!")
        text = "# This is a markDown Text"
        
        # Use the final state from streaming
        st.subheader("📝 Generated Blog")
        st.markdown(final_state["final"])
        
        # plan = final_state["plan"]
        # filename = f"{plan.blog_title}.md"
        # with open(filename, "w", encoding="utf-8") as f:
        #     f.write(text)

        # print("Markdown file saved as output.md")
