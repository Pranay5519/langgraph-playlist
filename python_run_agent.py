import sys
import subprocess
import uuid
import re
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# -----------------------------
# State
# -----------------------------

class AgentState(TypedDict):
    user_query: str
    generated_code: str
    execution_output: dict
    final_answer: str
    video_path: str
    retry_count: int
    max_retries: int
    error_history: list  # Track all errors


# -----------------------------
# LLM (Gemini 2.5 Flash)
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# Node 1 – generate python code
# -----------------------------

def code_generator_node(state: AgentState):
    prompt = f"""
You are a python code generator for Manim animations.

Write ONLY valid python code.
Do not add explanations.
Do not add markdown.
Import all necessary modules and constants.

User request:
{state["user_query"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "generated_code": response.content.strip(),
        "retry_count": 0,
        "error_history": []
    }


# -----------------------------
# Node 2 – Code Fixer (NEW!)
# -----------------------------

def code_fixer_node(state: AgentState):
    """Fix code based on error feedback"""
    
    last_error = state["execution_output"].get("stderr", "")
    previous_code = state["generated_code"]
    
    prompt = f"""
You are a python code debugger.

The following Manim code has an ERROR:


{previous_code}


ERROR:
{last_error}

Previous errors fixed:
{chr(10).join(state["error_history"][-3:]) if state["error_history"] else "None"}

Fix the code by:
1. Adding missing imports
2. Fixing undefined variables/constants
3. Correcting syntax errors
4. Ensuring all Manim constants are properly imported

Return ONLY the corrected python code.
No explanations. No markdown.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    new_error_history = state["error_history"] + [last_error]

    return {
        **state,
        "generated_code": response.content.strip(),
        "retry_count": state["retry_count"] + 1,
        "error_history": new_error_history
    }


# -----------------------------
# helpers for runner
# -----------------------------

def save_code_to_file(code: str) -> Path:
    path = Path("tmp")
    path.mkdir(exist_ok=True)

    file_path = path / f"{uuid.uuid4().hex}.py"
    file_path.write_text(code, encoding="utf-8")

    return file_path


def extract_manim_scene(code: str) -> str | None:
    """Extract first class that inherits from Scene."""
    pattern = r"class\s+(\w+)\s*\(\s*Scene\s*\)\s*:"
    match = re.search(pattern, code)
    if match:
        return match.group(1)
    return None


def find_generated_mp4() -> Path | None:
    """Find the most recently generated MP4 file in media/videos."""
    media_path = Path("media/videos")
    if not media_path.exists():
        return None
    
    mp4_files = list(media_path.rglob("*.mp4"))
    if not mp4_files:
        return None
    
    return max(mp4_files, key=lambda p: p.stat().st_mtime)


def run_manim_file(path: Path, scene_name: str):
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                "-pql",
                str(path),
                scene_name
            ],
            capture_output=True,
            text=True,
            timeout=180
        )

        video_path = None
        if result.returncode == 0:
            video_path = find_generated_mp4()

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "video_path": str(video_path) if video_path else None
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Manim rendering timed out",
            "returncode": -1,
            "video_path": None
        }


# -------------------------------------------------
# Node 3 – Code Runner
# -------------------------------------------------

def code_runner_node(state):
    code = state["generated_code"]
    path = save_code_to_file(code)

    scene_name = extract_manim_scene(code)

    if scene_name is not None:
        output = run_manim_file(path, scene_name)
    else:
        output = {
            "stdout": "",
            "stderr": "No Scene class found in generated code",
            "returncode": -1,
            "video_path": None
        }

    return {
        **state,
        "execution_output": output,
        "video_path": output.get("video_path", "")
    }


# -----------------------------
# Node 4 – Check if needs retry
# -----------------------------

def should_retry(state: AgentState) -> str:
    """Decide if we should retry or finish"""
    
    output = state["execution_output"]
    
    # Success case
    if output["returncode"] == 0:
        return "final_answer"
    
    # Failed but can retry
    if state["retry_count"] < state["max_retries"]:
        print(f"\n⚠️  Error detected. Retry {state['retry_count'] + 1}/{state['max_retries']}")
        return "code_fixer"
    
    # Failed and out of retries
    return "final_answer"


# -----------------------------
# Node 5 – final answer node
# -----------------------------

def final_answer_node(state: AgentState):
    out = state["execution_output"]

    if out["returncode"] != 0:
        answer = f"""❌ Failed after {state['retry_count']} retries.

Last Error:
{out['stderr']}

All errors encountered:
{chr(10).join(f"{i+1}. {err[:100]}..." for i, err in enumerate(state["error_history"]))}
"""
    else:
        answer = f"""✅ Animation generated successfully after {state['retry_count']} retries!

Video: {state['video_path']}

Errors fixed: {len(state['error_history'])}
"""

    return {
        **state,
        "final_answer": answer
    }


# -----------------------------
# Build graph with retry loop
# -----------------------------

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("code_fixer", code_fixer_node)
    graph.add_node("code_runner", code_runner_node)
    graph.add_node("final_answer", final_answer_node)

    # Set entry point
    graph.set_entry_point("code_generator")

    # Initial generation flow
    graph.add_edge("code_generator", "code_runner")
    
    # Conditional: retry or finish
    graph.add_conditional_edges(
        "code_runner",
        should_retry,
        {
            "code_fixer": "code_fixer",
            "final_answer": "final_answer"
        }
    )
    
    # Fix and retry
    graph.add_edge("code_fixer", "code_runner")
    
    # End
    graph.add_edge("final_answer", END)

    return graph.compile()


# -----------------------------
# Test
# -----------------------------

if __name__ == "__main__":
    graph = build_graph()
    user_query = """
Create a clean and simple Manim Community Python script that visually explains Logistic Regression in 2D using subtitles and voiceover.
in python Code only put ppython Code 
Important rules:
- Use only:
Dont use Scene Class 
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
and numpy.
- Do not modify any manim configuration.
- Do not use any advanced manim features.

Scene requirements:
- The scene class name must be LogisticRegressionScene.
- The scene must inherit from VoiceoverScene.
- Enable the GTTS voice service.

Layout and visual constraints (very important):
- Do NOT show any numeric labels on the X axis or Y axis.
- Do NOT show positive or negative tick labels.
- All visual elements must be positioned so that nothing overlaps.
- The dataset plot, sigmoid curve plot, title, subtitles, decision boundary, and any text must be clearly separated in space.
- Overlapping of any objects is not allowed.

Visualization requirements:
- Place a 2D Axes for the dataset on the LEFT side.
- Place a separate 2D Axes for the sigmoid curve on the RIGHT side.
- Add a clear title at the top: Logistic Regression.
- Add a subtitle text line at the bottom that updates during the animation.

Content and animation steps (each step must use voiceover and update subtitles):
1) Introduce logistic regression and show both axes.
2) Generate a small synthetic 2D dataset with two classes using different colors.
3) Display a straight decision boundary line.
4) Show a sigmoid curve that represents probability mapping.
5) Animate the decision boundary moving to a better separating position.
6) Conclude with a short explanation that logistic regression outputs probabilities.

Additional rules:
- Every major step must use a voiceover block.
- Subtitles must be updated to match the spoken text.
- No elements must overlap at any point in the animation.
- The output must be ONLY valid runnable Python code.
- Do not include explanations, markdown, or any text outside the Python code.
"""


    result = graph.invoke({
        "user_query": user_query,
        "generated_code": "",
        "execution_output": {},
        "final_answer": "",
        "video_path": "",
        "retry_count": 0,
        "max_retries": 3,  # Try up to 3 times to fix
        "error_history": []
    })
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(result["final_answer"])
    print("\n" + "="*60)
    print("FINAL CODE:")
    print("="*60)
    print(result["generated_code"])