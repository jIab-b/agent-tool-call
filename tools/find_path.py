import json
import os
import fnmatch
from .tool_base import Tool, register

def _scandir_recursive(path, pattern, search_type, matches):
    """
    Recursively finds files or directories using the efficient os.scandir(),
    skipping directories that cause PermissionError.
    """
    try:
        for entry in os.scandir(path):
            try:
                # Determine if the entry name matches the pattern
                is_match = fnmatch.fnmatch(entry.name, pattern)

                if entry.is_dir(follow_symlinks=False):
                    if search_type in ("all", "dir") and is_match:
                        matches.append(entry.path)
                    # Recurse into subdirectory
                    _scandir_recursive(entry.path, pattern, search_type, matches)
                elif entry.is_file(follow_symlinks=False):
                    if search_type in ("all", "file") and is_match:
                        matches.append(entry.path)
            except PermissionError:
                # Skip directories that cannot be accessed during recursion
                continue
    except PermissionError:
        # Skip top-level directory if it cannot be accessed
        pass

def _run(args: dict) -> str:
    """
    Finds files or directories matching a pattern starting from a root directory
    using a fast, pure-Python os.scandir() implementation.
    """
    pattern = args["pattern"]
    root = args.get("root", ".")
    search_type = args.get("type", "all")

    if os.path.abspath(root) == "/":
        return json.dumps({"error": "Searching from the filesystem root is not allowed."})

    matches = []
    try:
        _scandir_recursive(root, pattern, search_type, matches)
    except Exception as e:
        return json.dumps({"error": str(e)})
    
    return json.dumps(matches, ensure_ascii=False)

register(
    Tool(
        name="find_path",
        description="Finds files or directories by searching from a root path using a fast, pure-Python scandir-based implementation. Searching from the filesystem root is not permitted.",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The search pattern (e.g., 'my_file.txt' or '*.log')"},
                "root": {"type": "string", "description": "The root directory to start searching from. Defaults to the current project directory."},
                "type": {"type": "string", "description": "Type of path to find: 'file', 'dir', or 'all'. Defaults to 'all'."}
            },
            "required": ["pattern"],
        },
        run=_run,
    )
)