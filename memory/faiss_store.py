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

    def _get_model(self) -> SentenceTransformer:
        """Returns the singleton model instance, loading if needed."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def load(self):
        """Loads the index and metadata from disk."""
        model = self._get_model()
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            d = model.get_sentence_embedding_dimension()
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

    def save(self):
        """Saves the index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.metadata, f)

    async def ingest(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Ingests texts into the in-memory index."""
        model = self._get_model()
        vectors = model.encode(texts)
        
        start_index = self.index.ntotal
        self.index.add_with_ids(np.array(vectors).astype(np.float32), np.arange(start_index, start_index + len(texts)))
        
        for text, meta in zip(texts, metadatas):
            self.metadata.append({"text": text, "metadata": meta})

    async def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Queries the in-memory index."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        model = self._get_model()
        vec = model.encode([query])
        _, indices = self.index.search(np.array(vec).astype(np.float32), k)
        
        return [self.metadata[i] for i in indices[0] if 0 <= i < len(self.metadata)]
