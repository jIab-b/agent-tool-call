import os
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Memory

class FaissStore(Memory):
    def __init__(self, index_path: str = ".rag/index.faiss", meta_path: str = ".rag/meta.pkl", model_name: str = "all-MiniLM-L6-v2"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def _lazy_init(self):
        """Lazy load model and index to avoid startup cost."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        if self.index is None and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        elif self.index is None:
            d = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(d)

    async def ingest(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        self._lazy_init()
        vectors = self.model.encode(texts)
        if self.index.ntotal == 0:
            d = vectors.shape[1]
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

        combined_data = [{"text": text, "metadata": meta} for text, meta in zip(texts, metadatas)]
        self.index.add_with_ids(np.array(vectors).astype(np.float32), np.arange(len(self.metadata), len(self.metadata) + len(texts)))
        self.metadata.extend(combined_data)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    async def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        self._lazy_init()
        if self.index.ntotal == 0:
            return []
        vec = self.model.encode([query])
        D, I = self.index.search(np.array(vec).astype(np.float32), k)
        
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results
