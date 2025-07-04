from memory.faiss_store import FaissStore
from .tool_base import Tool, register

store = FaissStore()

async def _run(args: dict) -> str:
    await store.ingest(args["texts"], args["metadata"])
    return "Ingested successfully."

register(
    Tool(
        name="memory_ingest",
        description="Ingests texts and metadata into the vector store.",
        parameters={
            "type": "object",
            "properties": {
                "texts": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["texts", "metadata"],
        },
        run=_run,
    )
)