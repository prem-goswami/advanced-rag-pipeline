import threading
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        print("ℹ️ DocumentReranker initialized in lazy-load standby mode.")
        
        
    def _get_model(self):
        """Thread-safe helper to load the model exactly once when needed."""
        if self.model is None:
            with self._lock:
                if self.model is None:  # Double-check locking pattern
                    print("⏳ Lazy-Loading Cross-Encoder model into memory...")
                    self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    print("✅ Cross-encoder successfully loaded and operational.")
        return self.model

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        if not candidates:
            return []
        #fetch the model at runtime
        
        encoder_model = self._get_model()

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = encoder_model.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]