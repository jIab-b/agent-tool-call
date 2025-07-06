from dataclasses import dataclass
from .tool_base import Tool, register
import os

@dataclass
class FileWriteArgs:
    path: str
    content: str

def file_write(args: dict) -> str:
    """Writes content to a file, creating parent directories if needed."""
    path = args["path"]
    content = args["content"]
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File written successfully to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

register(
    Tool(
        name="file_write",
        description="Writes content to a specified file. Creates any missing parent directories and overwrites the file if it already exists.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
        run=file_write,
    )
)