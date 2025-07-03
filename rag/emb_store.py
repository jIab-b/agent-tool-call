import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, index_path: str = ".rag/index.faiss", meta_path: str = ".rag/meta.pkl"):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.index_path = index_path
        self.meta_path = meta_path
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = None
            self.metadata = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        vectors = self.model.encode(texts)
        if self.index is None:
            d = vectors.shape[1]
            self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(vectors).astype(np.float32))
        self.metadata.extend(metadatas)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = self.model.encode([query])
        D, I = self.index.search(np.array(vec).astype(np.float32), k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

# singleton instance
store = VectorStore()