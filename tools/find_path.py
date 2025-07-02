import json
import os
import fnmatch

from .tool_base import Tool, register


def _run(args: dict) -> str:
    """
    Finds files or directories matching a pattern starting from a root directory.
    """
    pattern = args["pattern"]
    root = args.get("root", os.path.expanduser("~"))
    search_type = args.get("type", "all")  # 'all', 'file', or 'dir'

    matches = []
    try:
        for current_root, dirs, files in os.walk(root):
            if search_type in ("all", "dir"):
                for d in fnmatch.filter(dirs, pattern):
                    matches.append(os.path.join(current_root, d))
            if search_type in ("all", "file"):
                for f in fnmatch.filter(files, pattern):
                    matches.append(os.path.join(current_root, f))
    except Exception as e:
        return json.dumps({"error": str(e)})
    
    return json.dumps(matches, ensure_ascii=False)


register(
    Tool(
        name="find_path",
        description="Finds files or directories by searching from a root path.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The search pattern (e.g., 'my_file.txt' or '*.log')"},
                "root": {"type": "string", "description": "The root directory to start searching from. Defaults to the user's home directory."},
                "type": {"type": "string", "description": "Type of path to find: 'file', 'dir', or 'all'. Defaults to 'all'."}
            },
            "required": ["pattern"],
        },
        run=_run,
    )
)