import json
import os
from pathlib import Path

from .tool_base import Tool, register


def _run(args: dict) -> str:
    """
    Lists files and directories at a given path.
    """
    path = Path(args.get("path", "."))
    recursive = args.get("recursive", False)
    
    if not path.exists() or not path.is_dir():
        return json.dumps({"error": "path is not a valid directory"})

    results = []
    try:
        if recursive:
            for root, dirs, files in os.walk(path):
                for name in files:
                    results.append(os.path.join(root, name))
                for name in dirs:
                    results.append(os.path.join(root, name))
        else:
            for entry in os.listdir(path):
                results.append(os.path.join(path, entry))
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps(results, ensure_ascii=False)


register(
    Tool(
        name="list_directory",
        description="Lists files and directories within a given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list. Defaults to the current directory."},
                "recursive": {"type": "boolean", "description": "Whether to list contents recursively. Defaults to false."}
            },
            "required": [],
        },
        run=_run,
    )
)