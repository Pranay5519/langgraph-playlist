import sys
import subprocess
from pathlib import Path


def run_manim_file(file_path: Path, scene_name: str) -> dict:
    """
    Runs a Manim scene and returns stdout, stderr and returncode.
    Uses the same Python environment (venv) as the caller.
    """

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "manim",
                "-pql",
                str(file_path),
                scene_name
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=file_path.parent
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Manim execution timed out",
            "returncode": -1
        }
out = run_manim_file(
    Path("manim_agent\main.py"),
    "LinearRegressionAnimation"
)
