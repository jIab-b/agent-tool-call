import json
from memory.faiss_store import FaissStore
from .tool_base import Tool, register

store = FaissStore()

async def _run(args: dict) -> str:
    results = await store.query(args["query"], args.get("k", 5))
    return json.dumps(results)

register(
    Tool(
        name="memory_query",
        description="Queries the vector store for similar texts.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        run=_run,
    )
)