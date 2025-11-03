import os
import pickle
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class RAGIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []  # list of text chunks corresponding to embedding rows

    def build(self, chunks: List[str]):
        if not chunks:
            raise ValueError("chunks must be non-empty")
        embs = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        self.metadata = chunks

    def save(self, index_path: str, meta_path: str):
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"model_name": self.model_name, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, index_path: str, meta_path: str):
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        model_name = data.get("model_name", "all-MiniLM-L6-v2")
        r = cls(model_name=model_name)
        r.index = faiss.read_index(index_path)
        r.metadata = data.get("metadata", [])
        return r

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        results: List[Tuple[str, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append((self.metadata[idx], float(dist)))
        return results
