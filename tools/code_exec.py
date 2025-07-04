import os
import importlib

from .tool_base import Tool, register

def get_sandbox():
    backend_name = os.getenv("SANDBOX_BACKEND", "local")
    if backend_name == "local":
        from sandbox.local import LocalSandbox
        return LocalSandbox()
    elif backend_name == "docker":
        from sandbox.docker import DockerSandbox
        return DockerSandbox()
    else:
        raise ValueError(f"Unknown sandbox backend: {backend_name}")

sandbox = get_sandbox()

def _run(args: dict) -> str:
    return sandbox.run_code(**args)

register(
    Tool(
        name="code_exec",
        description="Executes code in an isolated sandbox.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code to execute."},
                "image": {"type": "string", "description": "The Docker image to use (docker sandbox only)."},
                "timeout": {"type": "number", "default": 5.0},
                "memory": {"type": "string", "description": "Memory limit (e.g., '512m') for docker sandbox."},
                "cpus": {"type": "string", "description": "CPU limit (e.g., '0.5') for docker sandbox."},
                "network": {"type": "boolean", "default": False, "description": "Enable network access for docker sandbox."},
            },
            "required": ["code"],
        },
        run=_run,
    )
)