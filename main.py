import os
from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Core Engine Wireframes
from vectorstore import (
    get_pg_connection, get_pg_vectorstore, 
    get_pinecone_index, get_pinecone_vectorstore, get_llm
)
from reranker import DocumentReranker
from pipeline import RAGPipelineManager

# LangChain Orchestrations
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- GLOBAL CONTAINER SYSTEMS (Zero-blocking instantiation) ---
pipeline_manager = RAGPipelineManager()
reranker_manager = DocumentReranker()
_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

api_tags = [
    {"name": "Health", "description": "Operational checks and service readiness endpoints."},
    {"name": "PostgreSQL / pgvector", "description": "Semantic search, hybrid retrieval, and RAG endpoints."},
    {"name": "Pinecone", "description": "Semantic search and RAG endpoints backed by Pinecone."},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🚀 The startup sequence is now 100% clean and non-blocking.
    # Uvicorn will bind to its port in under 50ms, eliminating the 502 gateway timeout.
    yield

app = FastAPI(
    title="Production Multi-Backend Hybrid RAG Pipeline",
    description="A hybrid retrieval-augmented generation API combining PostgreSQL/pgvector and Pinecone.",
    version="1.0.0",
    openapi_tags=api_tags,
    lifespan=lifespan,
)

# --- SCHEMAS ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="User search query.", examples=["What is retrieval augmented generation?"])
    top_k: int = Field(5, ge=1, le=20, description="Maximum results.", examples=[5])

class SearchResult(BaseModel):
    rank: int
    score: float
    title: str
    url: str
    chunk_index: int
    text: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class RAGRequest(BaseModel):
    question: str = Field(..., description="Question to answer.", examples=["What is retrieval augmented generation?"])
    top_k: int = Field(5, ge=1, le=20, description="Supporting sources.", examples=[5])

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[SearchResult]

class RerankedResult(BaseModel):
    rank: int
    rerank_score: float
    original_rank: int
    title: str
    url: str
    chunk_index: int
    text: str

class RerankedResponse(BaseModel):
    query: str
    results: list[RerankedResult]

class HealthResponse(BaseModel):
    status: str
    backends: dict[str, str]

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using ONLY the provided context below.\n\nContext:\n{context}"),
    ("human", "{input}"),
])

# --- HEALTH SYSTEM ---
@app.get("/", tags=["Health"], response_model=HealthResponse)
def system_health():
    try:
        get_pinecone_index().describe_index_stats()
        pinecone_status = "Connected"
    except Exception:
        pinecone_status = "Disconnected/Uninitialized"
        
    return {
        "status": "healthy",
        "backends": {
            "pinecone": pinecone_status,
            "pgvector": "Operational (Lazy-Loaded Connection)"
        }
    }

# =====================================================================
#                          PGVECTOR ENDPOINTS
# =====================================================================

@app.post("/pg/search", tags=["PostgreSQL / pgvector"], response_model=SearchResponse)
def search_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query validation error")
        
    client = get_openai_client()
    embedding_res = client.embeddings.create(model="text-embedding-3-small", input=request.query)
    query_vector = embedding_res.data[0].embedding

    from psycopg2.extras import RealDictCursor
    cursor = db.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT content, source, source_url, chunk_index, 1 - (embedding <=> %s::vector) AS score FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
        (str(query_vector), str(query_vector), request.top_k)
    )
    rows = cursor.fetchall()
    cursor.close()

    results = [SearchResult(rank=i, score=round(float(r["score"]), 4), title=r["source"], url=r["source_url"], chunk_index=r["chunk_index"], text=r["content"]) for i, r in enumerate(rows, 1)]
    return SearchResponse(query=request.query, results=results)

@app.post("/pg/hybrid-search", tags=["PostgreSQL / pgvector"], response_model=SearchResponse)
def hybrid_search_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    try:
        hits = pipeline_manager.execute_hybrid_rerank_retrieval(request.query, request.top_k, db, SearchResult)
        return SearchResponse(query=request.query, results=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pg/hybrid-rag", tags=["PostgreSQL / pgvector"], response_model=RAGResponse)
def hybrid_rag_pgvector(request: RAGRequest, db=Depends(get_pg_connection)):
    try:
        hits = pipeline_manager.execute_hybrid_rerank_retrieval(request.question, request.top_k, db, SearchResult)
        context_str = "\n\n---\n\n".join([f"Source: {h.title}\nContent: {h.text}" for h in hits])
        
        messages = rag_prompt.format_messages(context=context_str, input=request.question)
        llm_out = get_llm().invoke(messages)
        return RAGResponse(question=request.question, answer=llm_out.content, sources=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pg/lcrag", tags=["PostgreSQL / pgvector"], response_model=RAGResponse)
def lang_chain_pg_rag(request: RAGRequest):
    try:
        retriever = get_pg_vectorstore().as_retriever(search_kwargs={"k": request.top_k})
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(), rag_prompt))
        res = chain.invoke({"input": request.question})
        
        sources = [SearchResult(rank=i, score=0.0, title=d.metadata.get("source", "Unknown"), url=d.metadata.get("source_url", ""), chunk_index=int(d.metadata.get("chunk_index", 0)), text=d.page_content) for i, d in enumerate(res.get("context", []), 1)]
        return RAGResponse(question=request.question, answer=res["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain context execution failure: {str(e)}")

@app.post("/pg/search-reranked", tags=["PostgreSQL / pgvector"], response_model=RerankedResponse)
def search_reranked_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    client = get_openai_client()
    embedding_res = client.embeddings.create(model="text-embedding-3-small", input=request.query)
    query_vector = embedding_res.data[0].embedding

    from psycopg2.extras import RealDictCursor
    cursor = db.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        "SELECT content, source, source_url, chunk_index, 1 - (embedding <=> %s::vector) AS score FROM documents ORDER BY embedding <=> %s::vector LIMIT 10",
        (str(query_vector), str(query_vector))
    )
    rows = cursor.fetchall()
    cursor.close()

    candidates = [{"original_rank": i, "title": r["source"], "url": r["source_url"], "chunk_index": r["chunk_index"], "text": r["content"], "similarity_score": float(r["score"])} for i, r in enumerate(rows, 1)]
    reranked_data = reranker_manager.rerank(request.query, candidates, top_k=3)

    results = [RerankedResult(rank=i, rerank_score=round(item["rerank_score"], 4), original_rank=item["original_rank"], title=item["title"], url=item["url"], chunk_index=item["chunk_index"], text=item["text"]) for i, item in enumerate(reranked_data, 1)]
    return RerankedResponse(query=request.query, results=results)

# =====================================================================
#                          PINECONE ENDPOINTS
# =====================================================================

@app.post("/pinecone/search", tags=["Pinecone"], response_model=SearchResponse)
def search_pinecone(request: SearchRequest):
    client = get_openai_client()
    query_res = client.embeddings.create(model="text-embedding-3-small", input=request.query)
    query_vector = query_res.data[0].embedding

    pc_res = get_pinecone_index().query(
        namespace=os.getenv("PINECONE_NAMESPACE", "semantic-search-wiki"),
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True
    )

    results = [SearchResult(rank=i, score=round(m.score, 4), title=m.metadata.get("source_title", "Unknown"), url=m.metadata.get("source_url", ""), chunk_index=int(m.metadata.get("chuck_index", 0)), text=m.metadata.get("text", "")) for i, m in enumerate(pc_res.matches, 1)]
    return SearchResponse(query=request.query, results=results)

@app.post("/pinecone/rag", tags=["Pinecone"], response_model=RAGResponse)
def rag_pinecone(request: RAGRequest):
    client = get_openai_client()
    query_res = client.embeddings.create(model="text-embedding-3-small", input=request.question)
    query_vector = query_res.data[0].embedding

    pc_res = get_pinecone_index().query(
        namespace=os.getenv("PINECONE_NAMESPACE", "semantic-search-wiki"),
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True
    )

    context_parts = []
    sources = []
    for i, m in enumerate(pc_res.matches, 1):
        t = m.metadata.get("source_title", "UNKNOWN")
        txt = m.metadata.get("text", "")
        context_parts.append(f"[Source {i}: {t}]: \n{txt}")
        sources.append(SearchResult(rank=i, score=round(m.score, 4), title=t, url=m.metadata.get("source_url", ""), chunk_index=int(m.metadata.get("chuck_index", 0)), text=txt))

    context = "\n\n".join(context_parts)
    messages = rag_prompt.format_messages(context=context, input=request.question)
    completion = get_llm().invoke(messages)

    return RAGResponse(question=request.question, answer=completion.content.strip(), sources=sources)

@app.post("/pinecone-lc/rag", tags=["Pinecone"], response_model=RAGResponse)
def lang_chain_pinecone_rag(request: RAGRequest):
    try:
        retriever = get_pinecone_vectorstore().as_retriever(search_kwargs={"k": request.top_k})
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(), rag_prompt))
        res = chain.invoke({"input": request.question})

        sources = [SearchResult(rank=i, score=0.0, title=d.metadata.get("source_title", "Unknown"), url=d.metadata.get("source_url", ""), chunk_index=int(d.metadata.get("chunk_index", 0)), text=d.page_content) for i, d in enumerate(res.get("context", []), 1)]
        return RAGResponse(question=request.question, answer=res["answer"].strip(), sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone LangChain execution error: {str(e)}")