import json
from pathlib import Path
from typing import List

from .tool_base import Tool, register


def _read_lines(path: Path, start: int | None, end: int | None) -> List[str]:
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    start = max((start or 1), 1)
    end = min((end or len(lines)), len(lines))
    return [f"{i+1} | {lines[i].rstrip()}" for i in range(start - 1, end)]


def _run(args: dict) -> str:
    p = Path(args["path"])
    if not p.exists() or p.is_dir():
        return json.dumps({"error": "file not found"})
    start = args.get("start_line")
    end = args.get("end_line")
    return json.dumps(_read_lines(p, start, end), ensure_ascii=False)


register(
    Tool(
        name="file_read",
        description="read a file (optionally line range)",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["path"],
        },
        run=_run,
    )
)