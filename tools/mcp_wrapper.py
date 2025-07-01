import json
from .tool_base import Tool, register


def _run(args: dict) -> str:
    """
    Prototype MCP wrapper.

    Simply returns the requested payload untouched.  In a full
    implementation you would invoke:
        use_mcp_tool(server_name=..., tool_name=..., arguments=...)
    via an async bridge and return the result.
    """
    return json.dumps({"notice": "mcp passthrough", **args}, ensure_ascii=False)


register(
    Tool(
        name="mcp_wrapper",
        description="proxy a call to a connected MCP server/tool",
        parameters={
            "type": "object",
            "properties": {
                "server_name": {"type": "string"},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object", "additionalProperties": True},
            },
            "required": ["server_name", "tool_name", "arguments"],
        },
        run=_run,
    )
)