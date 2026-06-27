from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self):
        print("⏳ Loading cross-encoder model...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-encoder ready.")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]