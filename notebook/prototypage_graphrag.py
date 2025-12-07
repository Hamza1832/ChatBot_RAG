# ===========================
# Imports
# ===========================
import psycopg
from psycopg import Cursor
import ollama
from pathlib import Path
import json

# ===========================
# Configuration
# ===========================
TRANS_FOLDER = Path(r"C:\Users\hamza\Desktop\ChatbotRag\ChatBot_RAG\data\TRANS_TXT")
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "llama3"
DB_CONN = "dbname=rag_chatbot user=postgres password=1803 host=localhost port=5432"

# ===========================
# Helpers
# ===========================
def read_and_filter(file_path: Path) -> list[str]:
    """Read a file and return filtered lines."""
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.read().split("\n")
    
    chunks = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("<")
    ]
    print(f"[read_and_filter] {file_path} -> {len(chunks)} chunks")
    return chunks


def create_embedding(text: str) -> list[float]:
    """Create embedding vector using Ollama."""
    try:
        emb = ollama.embeddings(EMBED_MODEL, text)["embedding"]
        print(f"[create_embedding] Length: {len(emb)}")
        return emb
    except Exception as e:
        print(f"[create_embedding] Failed for text: {text[:50]}... Error: {e}")
        return []


def extract_entities(text: str) -> list[str]:
    """Extract named entities using Ollama LLM."""
    prompt = f"""
    Extract named entities (people, organizations, locations, concepts)
    from the following text. Output JSON list only.

    TEXT:
    {text}
    """
    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=prompt,
            format="json"
        )
        entities = json.loads(response.get("response", "[]"))
        print(f"[extract_entities] {entities}")
        return entities
    except Exception as e:
        print(f"[extract_entities] Failed for text: {text[:50]}... Error: {e}")
        return []


def to_vector(vec: list[float]) -> str:
    """Convert list of floats to Postgres vector string."""
    return "[" + ",".join(map(str, vec)) + "]"

# ===========================
# DB Insert Functions
# ===========================
def save_chunk(text: str, embedding: list[float], cur: Cursor) -> int:
    if not embedding:
        print("[save_chunk] Empty embedding, skipping chunk.")
        return -1
    cur.execute(
        """
        INSERT INTO graph_chunks (chunk, embedding)
        VALUES (%s, %s::vector)
        RETURNING id
        """,
        (text, to_vector(embedding))
    )
    chunk_id = cur.fetchone()[0]
    print(f"[save_chunk] Inserted chunk ID: {chunk_id}")
    return chunk_id


def get_or_create_entity(name: str, cur: Cursor) -> int:
    cur.execute(
        """
        INSERT INTO graph_entities (name)
        VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,)
    )
    eid = cur.fetchone()[0]
    print(f"[get_or_create_entity] Entity '{name}' ID: {eid}")
    return eid


def link_chunk_entity(chunk_id: int, entity_id: int, cur: Cursor):
    if chunk_id == -1:
        return
    cur.execute(
        """
        INSERT INTO graph_edges (chunk_id, entity_id)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        (chunk_id, entity_id)
    )
    print(f"[link_chunk_entity] Linked chunk {chunk_id} -> entity {entity_id}")

# ===========================
# Similarity + Graph Expansion
# ===========================
def retrieve_context(query: str, k: int, cur: Cursor):
    emb = create_embedding(query)
    if not emb:
        print("[retrieve_context] Empty embedding for query")
        return []

    cur.execute(
        """
        SELECT c.id, c.chunk
        FROM graph_chunks c
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (to_vector(emb), k)
    )

    chunk_ids = [row[0] for row in cur.fetchall()]
    print(f"[retrieve_context] Retrieved chunk IDs: {chunk_ids}")

    if not chunk_ids:
        return []

    # Expand graph via shared entities
    cur.execute(
        """
        SELECT DISTINCT c.chunk
        FROM graph_edges e
        JOIN graph_edges e2 ON e.entity_id = e2.entity_id
        JOIN graph_chunks c ON e2.chunk_id = c.id
        WHERE e.chunk_id = ANY(%s)
        """,
        (chunk_ids,)
    )

    related_chunks = [r[0] for r in cur.fetchall()]
    print(f"[retrieve_context] Related chunks count: {len(related_chunks)}")
    return related_chunks

# ===========================
# Main
# ===========================
with psycopg.connect(DB_CONN) as conn:
    conn.autocommit = True

    with conn.cursor() as cur:

        # ------------------------
        # Schema
        # ------------------------
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("DROP TABLE IF EXISTS graph_edges")
        cur.execute("DROP TABLE IF EXISTS graph_entities")
        cur.execute("DROP TABLE IF EXISTS graph_chunks")

        cur.execute("""
        CREATE TABLE graph_chunks (
            id SERIAL PRIMARY KEY,
            chunk TEXT,
            embedding VECTOR(768)
        )
        """)

        cur.execute("""
        CREATE TABLE graph_entities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE
        )
        """)

        cur.execute("""
        CREATE TABLE graph_edges (
            chunk_id INT REFERENCES graph_chunks(id),
            entity_id INT REFERENCES graph_entities(id),
            UNIQUE (chunk_id, entity_id)
        )
        """)

        # ------------------------
        # Ingestion Pipeline
        # ------------------------
        files = list(TRANS_FOLDER.glob("*.txt"))
        print(f"[Main] Found {len(files)} files in {TRANS_FOLDER}")
        if not files:
            print("[Main] No files found, check TRANS_FOLDER path!")

        for file_path in files:
            print(f"[Main] Processing file: {file_path}")
            chunks = read_and_filter(file_path)

            for chunk in chunks:
                print(f"[Main] Processing chunk: {chunk[:50]}...")
                embedding = create_embedding(chunk)
                chunk_id = save_chunk(chunk, embedding, cur)

                entities = extract_entities(chunk)
                for ent in entities:
                    eid = get_or_create_entity(ent, cur)
                    link_chunk_entity(chunk_id, eid, cur)

        # ------------------------
        # Test Retrieval
        # ------------------------
        print("\n--- GraphRAG Context ---")
        context = retrieve_context("bonjour je veux de l'aide", 3, cur)
        for c in context:
            print("-", c)
