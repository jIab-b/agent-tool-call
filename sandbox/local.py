import subprocess
import tempfile
import textwrap
from pathlib import Path

from .base import Sandbox

class LocalSandbox(Sandbox):
    def run_code(
        self,
        code: str,
        *,
        image: str | None = None,  # ignored
        timeout: float = 5.0,
        memory: str | None = None,  # ignored
        cpus: str | None = None,  # ignored
        network: bool = False,  # ignored
    ) -> dict:
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
                return {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "exit_code": proc.returncode,
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired as e:
                return {
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or "Execution timed out",
                    "exit_code": -1,
                    "timed_out": True,
                }
