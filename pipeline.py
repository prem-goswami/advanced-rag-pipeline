import os
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from psycopg2.extras import RealDictCursor
from vectorstore import get_pg_connection

load_dotenv()

class RAGPipelineManager:
    def __init__(self):
        self.client = None
        self.bm25_index = None
        self.raw_chunks_lookup = {}

    def initialize_resources(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.hydrate_bm25_index()

    def hydrate_bm25_index(self):
        print("⏳ System boot: Hydrating global BM25 keyword index from PostgreSQL...")
        try:
            temp_conn = get_pg_connection()
            cursor = temp_conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'documents'
            );
            """)
            
            table_exists = cursor.fetchone()['exists']
            
            if not table_exists:
                print("ℹ️ 'documents' table does not exist yet. Skipping BM25 hydration until data is ingested.")
                cursor.close()
                temp_conn.close()
                return
            
            cursor.execute("SELECT id, content, source, source_url, chunk_index FROM documents")
            rows = cursor.fetchall()
            cursor.close()
            temp_conn.close()
            
            if not rows:
                print("ℹ️ 'documents' table is empty. BM25 standby mode active.")
                return

            tokenized_corpus = []
            for row in rows:
                chunk_id = row["id"]
                self.raw_chunks_lookup[chunk_id] = row
                tokenized_chunk = row["content"].lower().split()
                tokenized_corpus.append(tokenized_chunk)

            self.bm25_index = BM25Okapi(tokenized_corpus)
            print(f"Hybrid search engine ready: {len(rows)} documents cached in RAM.")
        except Exception as e:
            print(f"⚠️ BM25 Hydration skipped or failed (Expected if DB empty during initial setup): {e}")

    def execute_hybrid_rerank_retrieval(self, query: str, top_k: int, pg_conn, search_result_schema) -> list:
        # 1. Dense Search
        embedding_response = self.client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        query_vector = embedding_response.data[0].embedding

        cursor = pg_conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS similarity_score
            FROM documents ORDER BY embedding <=> %s::vector LIMIT 50
            """,
            (str(query_vector), str(query_vector)),
        )
        semantic_rows = cursor.fetchall()
        cursor.close()

        # 2. Sparse Search
        if not self.bm25_index:
            raise ValueError("BM25 index is uninitialized. Verify database collection records.")
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        for rank, row in enumerate(semantic_rows, 1):
            chunk_id = row["id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank))

        all_keys = list(self.raw_chunks_lookup.keys())
        all_bm25_matches = [(all_keys[idx], score) for idx, score in enumerate(bm25_scores)]
        all_bm25_matches.sort(key=lambda x: x[1], reverse=True)
        top_50_bm25 = all_bm25_matches[:50]

        for rank, (chunk_id, score) in enumerate(top_50_bm25, 1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (60.0 + rank))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_top_matches = sorted_rrf[:top_k]

        final_results = []
        for rank, (chunk_id, rrf_score) in enumerate(final_top_matches, 1):
            cached_row = self.raw_chunks_lookup[chunk_id]
            final_results.append(
                search_result_schema(
                    rank=rank,
                    score=round(rrf_score, 4),
                    title=cached_row["source"],
                    url=cached_row["source_url"],
                    chunk_index=cached_row["chunk_index"],
                    text=cached_row["content"],
                )
            )
        return final_results