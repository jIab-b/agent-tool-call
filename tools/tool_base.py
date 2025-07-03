import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# Optional runtime validation (no hard dependency)
try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None

@dataclass
class Tool:  # simple schema matching OpenAI-style functions
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON-schema-like
    run: Callable[[Dict[str, Any]], str]

class ToolRegistry:
    """A central registry for managing and validating tools."""
    def __init__(self):
        self._registry: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._registry[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a wrapped Tool that validates args against the JSON schema."""
        base = self._registry[name]

        def _validated_run(args: Dict[str, Any]) -> str:
            if jsonschema is not None:
                try:
                    jsonschema.validate(args, base.parameters)
                except Exception as e:
                    raise ValueError(f"args validation failed for {base.name}: {e}")
            return base.run(args)

        return Tool(base.name, base.description, base.parameters, _validated_run)

    def list_available(self) -> List[Tool]:
        return list(self._registry.values())

    def to_openai_def(self, t: Tool) -> Dict[str, Any]:
        """Export Tool metadata as an OpenAI function definition."""
        return {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }

    async def invoke(self, tool: Tool, args: Dict[str, Any]) -> str:
        """Run tool.run(), awaiting if it returns a coroutine."""
        res = tool.run(args)
        return await res if asyncio.iscoroutine(res) else res

# --- Singleton Instance and Global Wrappers ---
# This provides a single point of access and maintains backward compatibility
# with tool files that use the global `register` function.
_registry_instance = ToolRegistry()

def register(tool: Tool) -> None:
    _registry_instance.register(tool)

def get(name: str) -> Tool:
    return _registry_instance.get(name)

def get_global_registry() -> ToolRegistry:
    """Returns the singleton registry instance."""
    return _registry_instance

def list_available() -> List[Tool]:
    return _registry_instance.list_available()

def to_openai_def(t: Tool) -> Dict[str, Any]:
    return _registry_instance.to_openai_def(t)

async def invoke(tool: Tool, args: Dict[str, Any]) -> str:
    return await _registry_instance.invoke(tool, args)