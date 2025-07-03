import json
from .tool_base import Tool, register
from rag.emb_store import store

async def _run(args: dict) -> str:
    query = args["query"]
    k = args.get("k", 5)
    hits = store.query(query, k)
    return json.dumps(hits, ensure_ascii=False)

register(
    Tool(
        name="retrieve_context",
        description="retrieve top-k similar knowledge snippets from the vector store",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "query string for retrieval"
                },
                "k": {
                    "type": "integer",
                    "description": "number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        run=_run,
    )
)