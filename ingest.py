import os
import time
import psycopg2
import wikipedia
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = psycopg2.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", 5432)),
    dbname=os.getenv("PGDATABASE", "wikidb"),
    user=os.getenv("PGUSER", "pguser"),
    password=os.getenv("PGPASSWORD", "pass"),
    connect_timeout=10,
)
cursor = conn.cursor()

# =====================================================================
#Schema Setup
# =====================================================================
print("Setting up schema...")

cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# id is TEXT to support LangChain PGVectorStore compatibility
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        source TEXT,
        source_url TEXT,
        chunk_index INTEGER DEFAULT 0,
        embedding vector(1536)
    );
""")

# LangChain PGVectorStore requires this collection tracking table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS langchain_pg_collection (
        name VARCHAR,
        cmetadata JSON,
        uuid UUID PRIMARY KEY
    );
""")

# LangChain PGVectorStore requires this embedding table linked to collections
cursor.execute("""
    CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
        id VARCHAR PRIMARY KEY,
        collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
        embedding vector(1536),
        document VARCHAR,
        cmetadata JSON
    );
""")

conn.commit()
print("Schema ready.")

# =====================================================================
#Load Wikipedia Articles
# =====================================================================
wikipedia.set_user_agent(
    "WikiSemanticSearchDemo/1.0 (contact@example.com) Python-Wikipedia-Package"
)

topics = [
    "Artificial intelligence", "Machine learning", "Artificial neural network",
    "Natural language processing", "Computer vision", "Python (programming language)",
    "JavaScript", "Linux", "Cloud computing", "Docker (software)", "World War II",
    "American Revolutionary War", "Roman Empire", "Ancient Egypt", "Cold War",
    "Albert Einstein", "Isaac Newton", "Nikola Tesla", "Marie Curie", "Stephen Hawking",
    "Photosynthesis", "DNA", "Black hole", "Quantum mechanics", "Theory of relativity",
    "Association football", "Tennis", "Olympic Games", "FIFA World Cup", "Bitcoin",
    "Stock market", "Inflation", "Venture capital", "Gross domestic product", "Pizza",
    "Sushi", "Coffee", "Chocolate", "Bread", "Amazon rainforest", "Climate change",
    "Earthquake", "Volcano", "Ocean", "Leonardo da Vinci", "William Shakespeare",
    "Ludwig van Beethoven", "Pablo Picasso", "The Beatles", "Deep learning",
    "Data science", "Reinforcement learning",
]

# Resume support — skip already ingested topics
cursor.execute("SELECT DISTINCT source FROM documents;")
already_ingested = {row[0] for row in cursor.fetchall()}
if already_ingested:
    print(f"Resuming: {len(already_ingested)} topics already ingested, skipping them.")

articles = []
for topic in topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        if page.title in already_ingested:
            print(f"⏭  Skip (already ingested): {page.title}")
            continue
        articles.append({
            "title": page.title,
            "content": page.content,
            "url": page.url,
        })
        print(f"✓ Loaded: {page.title}")
        time.sleep(0.5)
    except Exception as e:
        print(f"✗ Skipped {topic}: {e}")

print(f"\nArticles to ingest this run: {len(articles)}")

# =====================================================================
# Chunk
# =====================================================================
def generate_overlapping_chunks(raw_articles, chunk_size=200, overlap=50):
    processed_chunks = []
    step_size = chunk_size - overlap
    for article in raw_articles:
        words = article["content"].split()
        chunk_count = 0
        for i in range(0, len(words), step_size):
            chunk = words[i: i + chunk_size]
            if len(chunk) < 30:
                continue
            processed_chunks.append({
                "id": f"{article['title'].replace(' ', '_')}-chunk-{chunk_count}",
                "text": " ".join(chunk),
                "source": article["title"],
                "source_url": article["url"],
                "chunk_index": chunk_count,
            })
            chunk_count += 1
    print(f"✅ {len(processed_chunks)} chunks created")
    return processed_chunks


chunks = generate_overlapping_chunks(articles)

if not chunks:
    print("Nothing to ingest. Exiting.")
    cursor.close()
    conn.close()
    exit(0)

# =====================================================================
# Embed and Insert
# =====================================================================
def embed_and_insert(chunks, batch_size=100):
    total = len(chunks)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            for j, embedding_data in enumerate(response.data):
                chunk = batch[j]
                cursor.execute(
                    """
                    INSERT INTO documents (id, content, source, source_url, chunk_index, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (
                        chunk["id"],
                        chunk["text"],
                        chunk["source"],
                        chunk["source_url"],
                        chunk["chunk_index"],
                        str(embedding_data.embedding),
                    ),
                )
            conn.commit()
            inserted += len(batch)
            print(f"Progress: {inserted}/{total} inserted")
            time.sleep(0.1)

        except Exception as e:
            print(f"⚠️  Error in batch {i}-{i+batch_size}: {e}")
            conn.rollback()
            time.sleep(5)
            continue

    print(f"\n✅ Done. {inserted} rows processed.")
    return inserted


total_inserted = embed_and_insert(chunks)

# =====================================================================
# Create vector index AFTER data is loaded (faster this way)
# =====================================================================
if total_inserted > 0:
    print("Building vector index...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS documents_embedding_idx
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    conn.commit()
    print("✅ Index created.")

# =====================================================================
#  Verify
# =====================================================================
cursor.execute("SELECT COUNT(*) FROM documents;")
print(f"\nTotal rows in database: {cursor.fetchone()[0]}")

cursor.execute("SELECT source, COUNT(*) FROM documents GROUP BY source ORDER BY source;")
print("\nChunks per article:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} chunks")

cursor.close()
conn.close()
print("\nIngestion complete. Database connection closed.")