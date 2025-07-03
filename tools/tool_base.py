from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# optional runtime validation (no hard dependency)
try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

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
    """Return a wrapped Tool whose run validates args against the JSON schema (when available)."""
    base = _registry[name]

    def _validated_run(args: Dict[str, Any]) -> str:
        if jsonschema is not None:
            try:
                jsonschema.validate(args, base.parameters)
            except Exception as e:
                raise ValueError(f"args validation failed for {base.name}: {e}")
        return base.run(args)

    # Return a shallow wrapper preserving metadata
    return Tool(base.name, base.description, base.parameters, _validated_run)


def list_available() -> List[Tool]:
    return list(_registry.values())

import asyncio

def to_openai_def(t: Tool) -> Dict[str, Any]:
    """Export Tool metadata as OpenAI function definition."""
    return {
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
    }

async def invoke(tool: Tool, args: Dict[str, Any]) -> str:
    """Run tool.run(), awaiting if it returns a coroutine."""
    res = tool.run(args)
    return await res if asyncio.iscoroutine(res) else res