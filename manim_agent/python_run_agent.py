import sys
import subprocess
import uuid
import re
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import Optional
load_dotenv()

class AgentState(TypedDict):
    user_query: str
    generated_code: str
    scene_name : str
    execution_output: dict
    final_answer: str
    video_path: str
    retry_count: int
    max_retries: int
    error_history: list  # Track all errors
    error_fixed : str


class CodeOutput(BaseModel):
    code: str
    fix_explanation: Optional[str] = Field(
        default=None,
        description="Explanation of how the error was fixed"
    )
    
    
class ExpandedPrompt(BaseModel):
    expanded_prompt: str
    
# -----------------------------
# LLM (Gemini 2.5 Flash)
# -----------------------------

llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0.7
)
google_llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


def system_prompt_expander(scene_name):
        
    system_prompt = f"""
    You are a Manim scene planner. Your job is to take a short user request and
    expand it into a detailed, technical Manim scene description that another AI
    will use to write Python code.

    Scene class name: {scene_name}

    Think like a Manim developer planning a scene. Your expanded prompt MUST include:
    1. OBJECTS — What Mobjects to create (Circle, Text, Cylinder, etc.) with exact colors using built-in constants (RED, BLUE, GREEN, DARK_BROWN etc.) or ManimColor("#RRGGBB") for custom colors
    2. POSITIONS — Where each object goes using move_to(), next_to(), to_edge() — NEVER use shift() for absolute positions
    3. ANIMATION ORDER — Step by step what happens and in what sequence (Create, Write, FadeIn, Transform, etc.)
    4. TIMING — How long each animation runs, where to self.wait()
    5. TEXT — Any labels or titles to display, their font sizes
    6. CAMERA — If 3D, what camera angle (phi, theta in degrees)

    ================================================================================
    SCREEN BOUNDARY RULES — STRICT
    ================================================================================
    The Manim screen is 16:9 ratio. Safe coordinate boundaries are:
    X axis: -6.5 to +6.5 (horizontal)
    Y axis: -3.5 to +3.5 (vertical)

    - ALL objects, text, and shapes MUST stay within these boundaries
    - Text font size must be 28 or smaller for body text, 40 max for titles
    - If placing multiple text lines, stack them vertically with 0.6–0.8 unit gaps
    - NEVER place anything beyond x=±6 or y=±3.5 — it will go off screen
    - Long text MUST be broken into shorter lines using line breaks or multiple Text objects
    - For subtitles or captions, always place at y=-3.0 (bottom) and keep font_size <= 24

    ================================================================================
    TIMING & PACING RULES
    ================================================================================
    - Keep animations snappy — default run_time=1.0 unless something needs emphasis
    - self.wait() should be 0.5 to 1.5 seconds max between steps
    - Do NOT use self.wait(3) or longer unless it is the final hold at the end
    - Total scene length should aim for 30–60 seconds, not longer
    - Voiceover text should be SHORT sentences — max 12 words per voiceover block
    - Each voiceover block should match exactly ONE visual action (one play call)

    ================================================================================
    OBJECT LIFECYCLE RULES — PREVENT OVERLAPPING
    ================================================================================
    - ALWAYS remove or fade out objects before showing new ones in the same area
    - Use this pattern: "first show X, then fade out X, then show Y"
    - Never have more than 3–4 objects visible on screen at the same time
    - When explaining multiple concepts sequentially:
    1. Show concept A (text + visual)
    2. Hold for voiceover duration
    3. FadeOut concept A completely
    4. Show concept B (text + visual)
    5. Repeat
    - For formulas or text that stays visible across steps, explicitly state:
    "keep the title visible throughout" OR "fade out the previous formula first"
    - Default assumption: if not told to keep something, it should be removed
    before the next object appears

    Example good planning:
    "Show title at top (keep visible). Show circle at center. Add radius line.
    Explain with voiceover. Fade out radius line. Add diameter line. Explain.
    Fade out diameter line. Show formula at y=2.0. Explain. Fade out formula.
    Show final conclusion text."

    Example bad planning (causes overlaps):
    "Show circle. Show radius. Show diameter. Show formula. Show chord."
    (Everything piles up on screen — messy and overlapping)
    
    ================================================================================
    TEXT & OVERLAP RULES
    ================================================================================
    - NEVER display two Text objects at the same Y position — they will overlap
    - Before adding new text, always FadeOut or remove the previous text first
    - Use VGroup to group related text and manage it together
    - Titles go at y=+3.0 (top), body text at y=0 (center), captions at y=-3.0 (bottom)
    - NEVER use Write() and FadeIn() on two different texts at the same time
    unless they are at clearly different Y positions (at least 0.8 units apart)
    - If showing a list of points, show them one at a time — not all at once

    ================================================================================
    ML / DL / DATA VISUALIZATION RULES
    ================================================================================
    - Use MAXIMUM 5 data points on any graph or chart — no cluttered plots
    - For neural networks: show maximum 3 layers, maximum 4 nodes per layer
    - Label axes clearly but keep axis labels short (1–3 words)
    - For loss curves: use 4–6 points only, smooth curve shape
    - For datasets: show 3–5 example points only — not a full scatter plot
    - Prefer simple NumberPlane or Axes with minimal gridlines
    - Avoid 3D graphs for ML concepts — use 2D always unless specifically asked
    - Color-code different data classes with clearly distinct colors (RED vs BLUE, never similar shades)
    - Always add a short Text label to explain what the graph shows, placed at top

    ================================================================================
    COLOR RULES
    ================================================================================
    When describing colors, always specify the hex code in parentheses.
    Example: "brown (#8B4513)", "forest green (#228B22)"
    Use built-in constants when possible: RED, BLUE, GREEN, YELLOW, WHITE, BLACK,
    ORANGE, PURPLE, PINK, GREY, TEAL, GOLD, MAROON

    ================================================================================
    POSITION RULES
    ================================================================================
    When describing positions, always say:
    "positioned at X units left/right, Y units up/down from center"
    Use move_to(), next_to(), to_edge() — NEVER shift() for absolute positions

    ================================================================================
    ANIMATION LANGUAGE
    ================================================================================
    When describing animations, use plain English order:
    "first appear, then move right, then fade out"
    Always state: object name → animation type → direction/target → timing

    Output a precise, technical, step-by-step scene script — not a story.
    Be specific with Manim class names and method names.

    {{format_instructions}}
    """
    return system_prompt
def prompt_expander_node(state: AgentState) -> dict:

    scene_name = state["scene_name"]
    user_query = state["user_query"]

    parser = PydanticOutputParser(pydantic_object=ExpandedPrompt)


    system_prompt = system_prompt_expander(scene_name)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_query}")
    ])

    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "user_query": user_query
    })

    return {
        **state,
        "user_query": response.expanded_prompt
    }
#---------------------------------------
# Node 2: Code Generator
#-------------------------------------

import os

def load_system_prompt() -> str:
    prompt_path = r"C:\Users\prana\Desktop\VS_CODE\Langraph\manim_agent\manim_agent_system_prompt_v2.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    

def code_generator_node(state: AgentState):
    scene_name = state["scene_name"]
    system_prompt = load_system_prompt()

    prompt = f"""
{system_prompt}

================================================================================
TASK
================================================================================
You are a Python code generator for Manim animations.

STRICT RULES:
- The class name MUST be exactly: {scene_name}
- Write ONLY valid Python code
- Do not add explanations
- Do not add markdown
- Import all necessary modules

User request:
{state["user_query"]}
"""
    structured_llm = google_llm.with_structured_output(CodeOutput)
    response = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "generated_code": response.code,
        "retry_count": 0,
        "max_retries": 3,
        "error_history": []
    }
 
# -----------------------------
# Node 3 – Check if needs retry
# -----------------------------


def code_fixer_node(state: AgentState):
    """Fix code based on error feedback"""

    if state["execution_output"]["returncode"] != 0:

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

Return ONLY valid corrected python code.
"""

        structured_llm = google_llm.with_structured_output(CodeOutput)
        response = structured_llm.invoke(prompt)

        new_error_history = state["error_history"] + [last_error]

        return {
            **state,
            "generated_code": response.code,
            "retry_count": state["retry_count"] + 1,
            "error_history": new_error_history,
            "error_fixed": response.fix_explanation
        }

    return state

# -----------------------------
# helpers for runner
# -----------------------------

def save_code_to_file(code: str, state: AgentState) -> Path:
    "save code to tmp file"
    path = Path("tmp")
    path.mkdir(exist_ok=True)
    scene_name = state['scene_name']
    file_path = path / f"{state['scene_name']}.py"
    file_path.write_text(code, encoding="utf-8")

    return file_path

def extract_manim_scene(code: str) -> str | None:
    pattern = r"class\s+(\w+)\s*\([^)]*Scene[^)]*\)\s*:"
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

from datetime import datetime
from pathlib import Path
import re


def save_error_log(state: dict, run_output: dict):
    """
    Append only:
    - File + line number
    - Exact line of code where error occurred
    - Final exception line
    - LLM fix explanation (if available)
    """

    log_path = Path("error_log.txt")
    stderr = run_output.get("stderr", "")

    if not stderr:
        return

    # Extract the actual error code line (starts with "> ")
    code_line_match = re.search(r">\s*\d+\s*│.*", stderr)
    code_line = code_line_match.group(0) if code_line_match else "Code line not found"

    # Clean weird box characters
    code_line = re.sub(r"[│┌└─]+", "", code_line).strip()

    # Extract final exception line (last non-empty line)
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    final_error_line = lines[-1] if lines else "Unknown error"

    # Extract file + line number
    file_line_match = re.search(r"(.*\.py:\d+)", stderr)
    file_line_info = file_line_match.group(1) if file_line_match else "Unknown location"

    # Get fix explanation from state (if exists)
    fix_explanation = state.get("error_fixed")

    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n========================================\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Scene: {state.get('scene_name', 'Unknown')}\n")
        f.write(f"{file_line_info}\n")
        f.write(f"{code_line}\n")
        f.write(f"{final_error_line}\n")

        # ✅ Append LLM explanation if available
        if fix_explanation:
            f.write(f"fix: {fix_explanation.strip()}\n")

    return log_path


def run_manim_file(path: Path, scene_name: str, state: AgentState):
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                "-pqh",
                str(path),
                scene_name
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=120
        )

        video_path = None
        if result.returncode == 0:
            video_path = find_generated_mp4()
        else:
            # 🔥 Save error log automatically
            save_error_log(state, {
                "stderr": result.stderr,
                "returncode": result.returncode
            })

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "video_path": str(video_path) if video_path else None
        }

    except subprocess.TimeoutExpired:
        timeout_output = {
            "stderr": "Manim rendering timed out",
            "returncode": -1
        }

        save_error_log(state, timeout_output)

        return {
            "stdout": "",
            "stderr": timeout_output["stderr"],
            "returncode": -1,
            "video_path": None
        }

# -------------------------------------------------
# Node 3 – Code Runner
# -------------------------------------------------

def code_runner_node(state):
    code = state["generated_code"]
    path = save_code_to_file(code , state=state)

    scene_name = state['scene_name']

    if scene_name is not None:
        output = run_manim_file(path, scene_name , state)
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
    
    
def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("prompt_expander", prompt_expander_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("code_fixer", code_fixer_node)
    graph.add_node("code_runner", code_runner_node)
    graph.add_node("final_answer", final_answer_node)

    # Set entry point
    graph.set_entry_point("prompt_expander")

    # Initial generation flow
    graph.add_edge("prompt_expander", "code_generator")
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
    return graph.compile(debug=True)