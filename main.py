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

# --- GLOBAL CONTAINER SYSTEMS ---
state = {}

api_tags = [
    {
        "name": "Health",
        "description": "Operational checks and service readiness endpoints.",
    },
    {
        "name": "PostgreSQL / pgvector",
        "description": "Semantic search, hybrid retrieval, and RAG endpoints backed by PostgreSQL and pgvector.",
    },
    {
        "name": "Pinecone",
        "description": "Semantic search and RAG endpoints backed by Pinecone.",
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Boot Sequence Execution
    state["pipeline"] = RAGPipelineManager()
    state["pipeline"].initialize_resources()
    state["reranker"] = DocumentReranker()
    state["openai_client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    yield
    # Shutdown steps go here if needed
    state.clear()

app = FastAPI(
    title="Production Multi-Backend Hybrid RAG Pipeline",
    description=(
        "A hybrid retrieval-augmented generation API that combines PostgreSQL/pgvector, "
        "Pinecone, BM25 reranking, and OpenAI generation for searchable Wikipedia-style knowledge retrieval."
    ),
    version="1.0.0",
    contact={
        "name": "Project Maintainer",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=api_tags,
    lifespan=lifespan,
)

# --- SCHEMAS (Unified Schema Patterns) ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="User search query to embed and retrieve against the vector index.", examples=["What is retrieval augmented generation?"])
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of results to return.", examples=[5])

class SearchResult(BaseModel):
    rank: int = Field(..., description="1-based rank of the result in the returned list.")
    score: float = Field(..., description="Similarity or reranking score used to order the result.")
    title: str = Field(..., description="Source title for the retrieved chunk.")
    url: str = Field(..., description="Source URL associated with the chunk.")
    chunk_index: int = Field(..., description="Index of the chunk within the original document.")
    text: str = Field(..., description="Retrieved passage text used to answer or summarize the query.")

class SearchResponse(BaseModel):
    query: str = Field(..., description="The original query sent by the user.")
    results: list[SearchResult] = Field(..., description="Ordered list of matching results.")

class RAGRequest(BaseModel):
    question: str = Field(..., description="User question to answer using retrieved context.", examples=["Explain how hybrid retrieval works."])
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of supporting sources to include.", examples=[5])

class RAGResponse(BaseModel):
    question: str = Field(..., description="The original question sent by the user.")
    answer: str = Field(..., description="Generated answer grounded in retrieved context.")
    sources: list[SearchResult] = Field(..., description="Source passages used to generate the answer.")

class RerankedResult(BaseModel):
    rank: int = Field(..., description="Final reranked position.")
    rerank_score: float = Field(..., description="Score assigned by the reranker.")
    original_rank: int = Field(..., description="Position before reranking.")
    title: str = Field(..., description="Source title for the reranked passage.")
    url: str = Field(..., description="Source URL associated with the passage.")
    chunk_index: int = Field(..., description="Chunk index within the original document.")
    text: str = Field(..., description="Passage text returned from the reranker.")

class RerankedResponse(BaseModel):
    query: str = Field(..., description="The original query sent by the user.")
    results: list[RerankedResult] = Field(..., description="Reranked result list.")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status.")
    backends: dict[str, str] = Field(..., description="Backend connectivity snapshot.")

# Prompts
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using ONLY the provided context below. Do not use outside knowledge.\n\nContext:\n{context}"),
    ("human", "{input}"),
])

# --- HEALTH SYSTEM ---
@app.get(
    "/",
    tags=["Health"],
    summary="Service health check",
    description="Returns a lightweight health snapshot for the API and its configured retrieval backends.",
    response_model=HealthResponse,
)
def system_health():
    try:
        pc_stats = get_pinecone_index().describe_index_stats()
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

@app.post(
    "/pg/search",
    tags=["PostgreSQL / pgvector"],
    summary="Semantic search with PostgreSQL / pgvector",
    description="Embeds the query with OpenAI, searches pgvector, and returns the top matching chunks.",
    response_model=SearchResponse,
)
def search_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query validation error")
        
    embedding_res = state["openai_client"].embeddings.create(model="text-embedding-3-small", input=request.query)
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

@app.post(
    "/pg/hybrid-search",
    tags=["PostgreSQL / pgvector"],
    summary="Hybrid search with BM25 and semantic retrieval",
    description="Combines vector similarity and BM25 scores using reciprocal rank fusion to return stronger matches.",
    response_model=SearchResponse,
)
def hybrid_search_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    try:
        hits = state["pipeline"].execute_hybrid_rerank_retrieval(request.query, request.top_k, db, SearchResult)
        return SearchResponse(query=request.query, results=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/pg/hybrid-rag",
    tags=["PostgreSQL / pgvector"],
    summary="Hybrid RAG answer with PostgreSQL / pgvector",
    description="Retrieves supporting chunks using hybrid search, then generates a grounded answer with OpenAI.",
    response_model=RAGResponse,
)
def hybrid_rag_pgvector(request: RAGRequest, db=Depends(get_pg_connection)):
    try:
        hits = state["pipeline"].execute_hybrid_rerank_retrieval(request.question, request.top_k, db, SearchResult)
        context_str = "\n\n---\n\n".join([f"Source: {h.title}\nContent: {h.text}" for h in hits])
        
        messages = rag_prompt.format_messages(context=context_str, input=request.question)
        llm_out = get_llm().invoke(messages)
        return RAGResponse(question=request.question, answer=llm_out.content, sources=hits)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/pg/lcrag",
    tags=["PostgreSQL / pgvector"],
    summary="LangChain RAG with PostgreSQL / pgvector",
    description="Uses LangChain retrieval and document stuffing to answer a question from the PostgreSQL vector store.",
    response_model=RAGResponse,
)
def lang_chain_pg_rag(request: RAGRequest):
    try:
        retriever = get_pg_vectorstore().as_retriever(search_kwargs={"k": request.top_k})
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(), rag_prompt))
        res = chain.invoke({"input": request.question})
        
        sources = [SearchResult(rank=i, score=0.0, title=d.metadata.get("source", "Unknown"), url=d.metadata.get("source_url", ""), chunk_index=int(d.metadata.get("chunk_index", 0)), text=d.page_content) for i, d in enumerate(res.get("context", []), 1)]
        return RAGResponse(question=request.question, answer=res["answer"], sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain context execution failure: {str(e)}")

@app.post(
    "/pg/search-reranked",
    tags=["PostgreSQL / pgvector"],
    summary="Semantic search with reranking",
    description="Performs vector search and reranks the top candidates for a more relevant final result list.",
    response_model=RerankedResponse,
)
def search_reranked_pgvector(request: SearchRequest, db=Depends(get_pg_connection)):
    embedding_res = state["openai_client"].embeddings.create(model="text-embedding-3-small", input=request.query)
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
    reranked_data = state["reranker"].rerank(request.query, candidates, top_k=3)

    results = [RerankedResult(rank=i, rerank_score=round(item["rerank_score"], 4), original_rank=item["original_rank"], title=item["title"], url=item["url"], chunk_index=item["chunk_index"], text=item["text"]) for i, item in enumerate(reranked_data, 1)]
    return RerankedResponse(query=request.query, results=results)

# =====================================================================
#                          PINECONE ENDPOINTS
# =====================================================================

@app.post(
    "/pinecone/search",
    tags=["Pinecone"],
    summary="Semantic search with Pinecone",
    description="Embeds the query and searches the Pinecone namespace for the most similar chunks.",
    response_model=SearchResponse,
)
def search_pinecone(request: SearchRequest):
    query_res = state["openai_client"].embeddings.create(model="text-embedding-3-small", input=request.query)
    query_vector = query_res.data[0].embedding

    pc_res = get_pinecone_index().query(
        namespace=os.getenv("PINECONE_NAMESPACE", "semantic-search-wiki"),
        vector=query_vector,
        top_k=request.top_k,
        include_metadata=True
    )

    results = [SearchResult(rank=i, score=round(m.score, 4), title=m.metadata.get("source_title", "Unknown"), url=m.metadata.get("source_url", ""), chunk_index=int(m.metadata.get("chuck_index", 0)), text=m.metadata.get("text", "")) for i, m in enumerate(pc_res.matches, 1)]
    return SearchResponse(query=request.query, results=results)

@app.post(
    "/pinecone/rag",
    tags=["Pinecone"],
    summary="RAG answer with Pinecone",
    description="Retrieves Pinecone matches and generates a grounded answer with the retrieved context.",
    response_model=RAGResponse,
)
def rag_pinecone(request: RAGRequest):
    query_res = state["openai_client"].embeddings.create(model="text-embedding-3-small", input=request.question)
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

@app.post(
    "/pinecone-lc/rag",
    tags=["Pinecone"],
    summary="LangChain RAG with Pinecone",
    description="Uses LangChain retrieval with Pinecone as the backing vector store.",
    response_model=RAGResponse,
)
def lang_chain_pinecone_rag(request: RAGRequest):
    try:
        retriever = get_pinecone_vectorstore().as_retriever(search_kwargs={"k": request.top_k})
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(get_llm(), rag_prompt))
        res = chain.invoke({"input": request.question})

        sources = [SearchResult(rank=i, score=0.0, title=d.metadata.get("source_title", "Unknown"), url=d.metadata.get("source_url", ""), chunk_index=int(d.metadata.get("chunk_index", 0)), text=d.page_content) for i, d in enumerate(res.get("context", []), 1)]
        return RAGResponse(question=request.question, answer=res["answer"].strip(), sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone LangChain execution error: {str(e)}")