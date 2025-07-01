import glob
import json
import os
import re
from pathlib import Path

from .tool_base import Tool, register


def _run(args: dict) -> str:
    pattern = args["regex"]
    root = args.get("path", ".")
    max_hits = int(args.get("max_hits", 50))
    flags = re.IGNORECASE if args.get("ignore_case") else 0

    hits = []
    for f in glob.iglob(f"{root}/**/*", recursive=True):
        if not os.path.isfile(f):
            continue
        try:
            with open(f, "r", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    if re.search(pattern, line, flags=flags):
                        hits.append(
                            {"file": str(Path(f).resolve()), "line": lineno, "text": line.rstrip()}
                        )
                        if len(hits) >= max_hits:
                            return json.dumps(hits, ensure_ascii=False)
        except Exception:
            # silently skip unreadable files
            pass
    return json.dumps(hits, ensure_ascii=False)


register(
    Tool(
        name="file_search",
        description="regex search across workspace files",
        parameters={
            "type": "object",
            "properties": {
                "regex": {"type": "string", "description": "Python regex pattern"},
                "path": {"type": "string", "description": "directory root to search"},
                "max_hits": {"type": "integer", "description": "truncate results", "default": 50},
                "ignore_case": {"type": "boolean", "default": False},
            },
            "required": ["regex"],
        },
        run=_run,
    )
)