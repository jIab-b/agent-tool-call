from dataclasses import dataclass
from typing import Any, Callable, Dict, List

@dataclass
class Tool:  # simple schema matching OpenAI-style functions
    name: str
    description: str
    parameters: Dict[str, Any]         # JSON-schema-like
    run: Callable[[Dict[str, Any]], str]

_registry: Dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _registry[tool.name] = tool


def get(name: str) -> Tool:
    return _registry[name]


def list_available() -> List[Tool]:
    return list(_registry.values())