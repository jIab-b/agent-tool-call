import json
import subprocess
import textwrap
import tempfile
from pathlib import Path

from .tool_base import Tool, register


def _run(args: dict) -> str:
    """
    Execute short Python code in a temporary sandbox (subprocess with timeout).
    Returns stdout / stderr (truncated).
    """
    code = args["code"]
    timeout = float(args.get("timeout", 2.0))
    char_limit = int(args.get("char_limit", 1024))

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "snippet.py"
        script.write_text(textwrap.dedent(code))
        try:
            proc = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            out = "Execution timed out"
    return json.dumps(out[:char_limit], ensure_ascii=False)


register(
    Tool(
        name="sandbox",
        description="executes short python code in a sandboxed subprocess",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout": {"type": "number", "default": 2.0},
                "char_limit": {"type": "integer", "default": 1024},
            },
            "required": ["code"],
        },
        run=_run,
    )
)