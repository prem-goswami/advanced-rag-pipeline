import os
import psycopg2
from psycopg2.extras import RealDictCursor
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore
from langchain_pinecone import PineconeVectorStore

# Shared OpenAI Layer
def get_openai_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("OPENAI_API_KEY")
    )

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --- POSTGRESQL / PGVECTOR CLIENTS ---
def get_pg_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", 5432)),
        dbname=os.getenv("PGDATABASE", "wikidb"),
        user=os.getenv("PGUSER", "pguser"),
        password=os.getenv("PGPASSWORD", "pass"),
        connect_timeout=3
    )

def get_pg_vectorstore():
    connection_string = f"postgresql+psycopg://{os.getenv('PGUSER', 'pguser')}:{os.getenv('PGPASSWORD', 'pass')}@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', 5432)}/{os.getenv('PGDATABASE', 'wikidb')}"
    engine = PGEngine.from_connection_string(connection_string)
    return PGVectorStore.create_sync(
        engine=engine,
        table_name="documents",
        embedding_service=get_openai_embeddings(),
        content_column="content",
        embedding_column="embedding",
        id_column="id",
    )

# --- PINECONE CLIENTS ---
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME", "semantic-search-wiki"))

def get_pinecone_vectorstore():
    return PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=get_openai_embeddings(),
        text_key="text",
        namespace=os.getenv("PINECONE_NAMESPACE", "semantic-search-wiki"),
    )