# Advanced RAG & Semantic Search Pipeline

A production-grade retrieval system demonstrating the full spectrum of modern AI search architecture 
from raw vector similarity to hybrid keyword fusion to neural reranking across two independent vector backends.

Built over **2,974 chunks** from **52 Wikipedia articles** with **8 endpoints** across **2 vector backends**.

**[Live Demo](https://advanced-rag-pipeline-production.up.railway.app)** · **[API Docs](https://advanced-rag-pipeline-production.up.railway.app/docs)**

---

## What This Demonstrates

Most RAG tutorials show one retrieval method. This project implements eight and documents exactly when each one wins and why.

| Endpoint | Backend | Method | Best For |
| :--- | :--- | :--- | :--- |
| `POST /pg/search` | pgvector | Dense semantic search | Conceptual queries |
| `POST /pg/hybrid-search` | pgvector | BM25 + RRF fusion | Technical keywords + intent |
| `POST /pg/hybrid-rag` | pgvector | Hybrid retrieval + generation | Grounded Q&A with precision |
| `POST /pg/lcrag` | pgvector + LangChain | Orchestrated RAG | Framework comparison |
| `POST /pg/search-reranked` | pgvector | Dense + cross-encoder reranking | Maximum retrieval precision |
| `POST /pinecone/search` | Pinecone | Dense semantic search | Managed cloud vector search |
| `POST /pinecone/rag` | Pinecone | RAG with citation enforcement | Grounded Q&A with sources |
| `POST /pinecone-lc/rag` | Pinecone + LangChain | Orchestrated RAG | Framework comparison |
| `GET /` | — | Health check | Backend connectivity status |

---
**Query Pipeline (per request)**
Query → Embed → Retrieve → [BM25 Fusion] → [Rerank] → [Generate] → Response

## Key Findings

These are empirical results from testing not claims from documentation.

**Pinecone vs pgvector are functionally equivalent for semantic quality**
Same embedding model + same cosine metric = scores converging to 4 decimal places.
- WWII query: Pinecone `0.6336` vs pgvector `0.6335`
- Photosynthesis query: Pinecone `0.5373` vs pgvector `0.5373`

The choice between them is operational, not algorithmic.

**Hybrid search outperforms pure semantic on exact technical terms**
Query: `"backpropagation gradient descent"`
- Pure semantic → Deep Learning article, historical overview chunk
- Hybrid search → Neural Network article, algorithmic explanation chunk

BM25 surfaced the precise term match that semantic similarity missed.

**Similarity ≠ Relevance the reranker proves it**
Query: `"How does backpropagation work in neural networks?"`
- pgvector ranked the correct algorithmic chunk at position **6**
- Cross-encoder reranked it to position **1**
- The chunk pgvector ranked 1st was demoted to position 2

**RAG graceful degradation works**
Query: `"What is the fastest animal on earth?"` — not in the knowledge base.
Response: `"I don't have enough information in my knowledge base to answer this."`
No hallucination. Knowledge boundary enforced via prompt.

---

## Architecture
<pre>
                                 User Query
                                     │
                          ┌──────────▼──────────┐
                          │    FastAPI Layer    │
                          │ (API Routing & DI)  │
                          └──────────┬──────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ pgvector Engine │         │ Pinecone Engine │         │ LangChain Layer │
├─────────────────┤         ├─────────────────┤         ├─────────────────┤
│ • Dense Search  │         │ • Dense Search  │         │ • PG VectorStore│
│ • Hybrid Search │         │ • Native RAG    │         │ • PC VectorStore│
│ • Reranked List │         │                 │         │ • Structured RAG│
│ • Custom RAG    │         │                 │         │                 │
└────────┬────────┘         └─────────────────┘         └─────────────────┘
         │
         └───────────────────────────┐
                                     ▼
                    ┌────────────────────────────────┐
                    │     Custom Processing Core     │
                    ├────────────────────────────────┤
                    │ • RAGPipelineManager           │
                    │   - Sparse BM25 Retrieval      │
                    │   - Reciprocal Rank Fusion     │
                    │ • DocumentReranker             │
                    │   - Cross-Encoder Re-scoring   │
                    └────────────────────────────────┘
</pre>

**Ingestion Pipeline (one-time setup)**

Wikipedia (52 articles)

→ Chunker (200 words, 50-word overlap)

→ OpenAI text-embedding-3-small

→ pgvector (PostgreSQL HNSW index) + Pinecone (serverless)

---



## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Pydantic v2 |
| Embedding Model | OpenAI text-embedding-3-small (1536 dimensions) |
| Generation Model | GPT-4o-mini (temperature 0.2) |
| Managed Vector DB | Pinecone (serverless, AWS us-east-1) |
| Self-hosted Vector DB | PostgreSQL + pgvector (HNSW index, cosine) |
| Sparse Retrieval | BM25 via rank-bm25 |
| Rank Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG Orchestration | LangChain (create_retrieval_chain) |
| Containerisation | Docker |
| Deployment | Railway |

---

## Retrieval Methods Explained

### Dense Semantic Search (`/pg/search`, `/pinecone/search`)
Embeds the query using OpenAI text-embedding-3-small and retrieves semantically similar chunks via cosine similarity. Handles intent, synonyms, and paraphrasing. Struggles with exact technical identifiers and proper nouns.

### Hybrid Search — BM25 + Reciprocal Rank Fusion (`/pg/hybrid-search`)
Combines two independent retrieval signals:
- **Dense** (pgvector): semantic similarity via embedding vectors
- **Sparse** (BM25): keyword frequency and rarity via TF-IDF statistics

Fused using **Reciprocal Rank Fusion** — not linear combination. BM25 scores are unbounded while cosine similarity is 0-1. Averaging them directly means BM25 dominates by scale. 
RRF discards raw scores and uses only rank position:
RRF_score = 1/(60 + rank_semantic) + 1/(60 + rank_BM25)
A chunk appearing in both lists accumulates scores from both — naturally surfacing high-consensus results regardless of score scale incompatibility. The BM25 index (2,974 chunks) is pre-loaded into RAM at server startup for sub-millisecond scoring. Metadata cached in an O(1) dictionary eliminates per-result database round-trips.

### Cross-Encoder Reranking (`/pg/search-reranked`)
After retrieving top-10 candidates via pgvector, the cross-encoder scores each `(query, chunk)` pair jointly — the transformer attention mechanism sees query tokens attending to document tokens simultaneously. Produces relevance scores rather than similarity scores.

Two-stage pattern:
1. **Stage 1** — bi-encoder (pgvector): top-10 candidates in milliseconds. Optimises for recall.
2. **Stage 2** — cross-encoder: precise reranking of 10 candidates. Optimises for precision.

The model (`ms-marco-MiniLM-L-6-v2`) is loaded at module level — ~90MB weight matrix loads once at startup, reused across all requests.

### Hybrid RAG (`/pg/hybrid-rag`)
The most complete pipeline: BM25 + RRF fusion retrieves the highest-precision candidate set, cross-encoder reranks to top-3, then GPT-4o-mini generates a grounded answer with citation enforcement. Combines the precision advantages of hybrid retrieval and neural reranking before generation.

### RAG with Citation Enforcement (`/pinecone/rag`)
Retrieved chunks formatted with `[Source N: Title]` labels and injected into a strict prompt. Temperature 0.2 keeps responses within the context boundary. Explicit fallback instruction enforces graceful degradation when context is insufficient.

### LangChain Orchestration (`/pg/lcrag`, `/pinecone-lc/rag`)
The same RAG pipeline implemented using LangChain's `create_retrieval_chain` and `create_stuff_documents_chain`. Side-by-side comparison with the manual implementation demonstrates what the framework abstracts — and validates that answer quality is equivalent.

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- OpenAI API key
- Pinecone API key

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/advanced-rag-pipeline
cd advanced-rag-pipeline
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_NAMESPACE=semantic-search-wiki
```

### 5. Spin up pgvector with Docker
```bash
docker run -d \
  --name pgvector-wiki \
  -e POSTGRES_PASSWORD=pgpass \
  -e POSTGRES_USER=pguser \
  -e POSTGRES_DB=wikidb \
  -p 5432:5432 \
  ankane/pgvector
```

Verify the container is running:
```bash
docker ps
```

Connect and enable the pgvector extension:
```bash
docker exec -it pgvector-wiki psql -U pguser -d wikidb
```

Inside psql:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extversion FROM pg_extension WHERE extname = 'vector';
-- Expected: 0.5.1 or higher

CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    content     TEXT,
    source      VARCHAR(500),
    source_url  VARCHAR(1000),
    chunk_index INT,
    embedding   vector(1536)
);

CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
\q
```

### 6. Ingest data
```bash
python ingest.py
```

This loads 52 Wikipedia articles, chunks them into 2,974 segments, generates embeddings via OpenAI, and upserts to both pgvector and Pinecone simultaneously. Takes approximately 5-8 minutes.

> **Railway deployment note:** After deploying to Railway, run the ingestion script from the Railway console:
> ```bash
> python ingest.py
> ```
> The pgvector database will be provisioned by Railway's PostgreSQL plugin. Ensure all environment variables are set in the Railway dashboard before running.

### 7. Start the API server
```bash
uvicorn main:app --reload
```

API live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---


## Design Decisions

**Why HNSW over IVFFlat?**
HNSW builds its graph at insert time — slower inserts, faster queries. For a read-heavy workload where you insert once and query many times, HNSW is the correct index. IVFFlat requires ANALYZE after bulk inserts and performs better on write-heavy workloads.

**Why RRF over linear score combination?**
BM25 scores are unbounded. Cosine similarity is 0-1. Averaging them directly means BM25 dominates by numeric scale, not retrieval quality. RRF uses only rank position — making all retrieval methods equally weighted regardless of their underlying scoring scales.

**Why 200-word chunks with 50-word overlap?**
200 words provides sufficient semantic context for a meaningful embedding without forcing the vector to represent too many concepts simultaneously. 50-word overlap ensures sentences split at chunk boundaries appear fully in at least one chunk — preserving semantic continuity at boundaries.

**Why retrieve top-10 before reranking to top-3?**
In testing, the most relevant chunk was at pgvector position 6. Retrieving only top-5 before reranking would have missed it entirely. A wider candidate pool gives the cross-encoder more material — its value is in surfacing results that similarity scoring underrated.

**Why lazy-load backends?**
Heavy components (CrossEncoder, pgvector connection, Pinecone client) are initialised on first use rather than at import time. This keeps server startup under 50ms, eliminates Railway 502 gateway timeouts, and allows the health endpoint to respond immediately while backends initialise in the background.

---

---

## What I Learned Building This

**The choice of retrieval backend is operational, not algorithmic.** Pinecone and pgvector return identical results when using the same embedding model. 
The decision framework is infrastructure existing stack, scale requirements, team expertise. Choosing Pinecone because it "sounds more AI" is not a reason.

**Similarity is not relevance.** A chunk ranked 6th by cosine similarity was the most relevant answer to the query. 
Vector search optimises for geometric proximity in embedding space. Cross-encoder reranking optimises for actual query-document relevance. 
They are correlated but not identical. That gap matters in production.

**Hybrid search matters most at the edges.** For broad conceptual queries, pure semantic search performs well. 
The BM25 signal becomes critical exactly when users search for something specific a model name, an algorithm, a technical identifier. 
Production search systems serve both types of queries.

---

## Author

**Prem Goswami** — AI Engineer
[LinkedIn](www.linkedin.com/in/prempurigoswami) · [GitHub](https://github.com/prem-goswami)
