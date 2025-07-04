from typing import Protocol, runtime_checkable, List, Dict, Any

@runtime_checkable
class Memory(Protocol):
    async def ingest(self, texts: List[str], metadata: List[Dict[str, Any]]):
        ...

    async def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        ...