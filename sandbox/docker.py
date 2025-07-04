import subprocess
import tempfile
import textwrap
from pathlib import Path

from .base import Sandbox

class DockerSandbox(Sandbox):
    def run_code(
        self,
        code: str,
        *,
        image: str | None = "python:3.11-slim",
        timeout: float = 5.0,
        memory: str | None = "512m",
        cpus: str | None = "0.5",
        network: bool = False,
    ) -> dict:
        with tempfile.TemporaryDirectory() as td:
            host_path = Path(td)
            container_path = "/workspace"
            script_path = host_path / "snippet.py"
            script_path.write_text(textwrap.dedent(code))

            cmd = [
                "docker", "run", "--rm",
                "--volume", f"{host_path.resolve()}:{container_path}:rw",
                "--workdir", container_path,
                "--memory", memory,
                "--cpus", cpus,
                "--pids-limit", "128",
            ]
            if not network:
                cmd.append("--network=none")
            
            cmd.extend([image, "python", "snippet.py"])

            try:
                proc = subprocess.run(
                    cmd,
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
                stdout = e.stdout.decode(errors="ignore") if e.stdout else ""
                stderr = e.stderr.decode(errors="ignore") if e.stderr else "Execution timed out"
                return {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": -1,
                    "timed_out": True,
                }