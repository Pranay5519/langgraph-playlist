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
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

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
from pydantic import BaseModel

class CodeOutput(BaseModel):
    code: str
    
    

# -----------------------------
# LLM (Gemini 2.5 Flash)
# -----------------------------

llm = ChatOllama(
    model="qwen3:latest",
    temperature=0.7
)

# -----------------------------
# Node 1 – generate python code
# -----------------------------

def code_generator_node(state: AgentState):
    scene_name = state["scene_name"]
    parser = PydanticOutputParser(pydantic_object=CodeOutput)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
You are a python code generator for Manim animations.

STRICT RULE:
The class name MUST be exactly: {scene_name}

Write ONLY valid python code.
Do not add explanations.
Do not add markdown.
Import all necessary modules.

{{format_instructions}}
"""
        ),
        ("human", "{user_query}")
    ])
    prompt = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | llm | parser   # ✅ plain ChatOllama

    response = chain.invoke({
        "user_query": state["user_query"]
    })

    return {
        **state,
        "generated_code": response.code,
        "retry_count": 0,
        "max_retries": 3,
        "error_history": []
    }
    
    
# -----------------------------
# Node 2 – Code Fixer (NEW!)
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

        structured_llm = llm.with_structured_output(CodeOutput)
        response = structured_llm.invoke(prompt)

        new_error_history = state["error_history"] + [last_error]

        return {
            **state,
            "generated_code": response.code,
            "retry_count": state["retry_count"] + 1,
            "error_history": new_error_history
        }

    return state

# -----------------------------
# helpers for runner
# -----------------------------

def save_code_to_file(code: str , state: AgentState) -> Path:
    "save code to tmp file"
    path = Path("tmp")
    path.mkdir(exist_ok=True)

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
        encoding="utf-8",   # ADD THIS
        errors="ignore",    # OPTIONAL SAFETY
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
    path = save_code_to_file(code , state=state)

    scene_name = state["scene_name"]

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

    return  graph.compile(debug=True)
# graph = build_graph()
# state = {
#     "user_query": "Implement NeuralNets working  using manim also keep voiceovers and better animation the ooutput video must be of atleast 50 seconds , and also make sure the elements are not overlapping each other",
#     "generated_code": "",
#     "scene_name" : "LogisticRegressionScene",
#     "execution_output": {},
#     "final_answer": "",
#     "video_path": "",
#     "retry_count": 0,
#     "max_retries": 3,
#     "error_history": []
# }

# response = graph.invoke(state)
# print(response["scene_name"], "\n")
# print(response["generated_code"])
