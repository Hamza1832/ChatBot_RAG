import json
from pathlib import Path
from typing import List, Dict, Any

import psycopg
from psycopg import Cursor
import ollama

# optional Neo4j (knowledge graph) integration
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False

# --------------------------
# Configuration
# --------------------------
TRANS_FOLDER = Path(r"C:\Users\hamza\Desktop\ChatbotRag\ChatBot_RAG\data\TRANS_TXT")
EMBED_MODEL = "embeddinggemma"
LLM_MODEL = "llama3"

DB_CONN = "dbname=rag_chatbot user=postgres password=1803 host=localhost port=5432"

# Neo4j optional config (only used if available and configured)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # change as needed

# embedding dimension used in DB vector column
EMBED_DIM = 768

# --------------------------
# Utilities
# --------------------------
def read_and_filter(file_path: Path) -> List[str]:
    """Read a file and return filtered lines/chunks."""
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    chunks = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("<")]
    print(f"[read_and_filter] {file_path.name} -> {len(chunks)} chunks")
    return chunks

def create_embedding(text: str) -> List[float]:
    """Create embedding vector using Ollama."""
    try:
        resp = ollama.embeddings(EMBED_MODEL, text)
        emb = resp.get("embedding") or resp.get("data")  # try common keys
        if not emb:
            raise ValueError("No embedding returned")
        return emb
    except Exception as e:
        print(f"[create_embedding] Error: {e}")
        return []

def to_pgvector_string(vec: List[float]) -> str:
    """Convert float list to pgvector literal string format [0.1,0.2,...]."""
    return "[" + ",".join(map(str, vec)) + "]"

def extract_entities_via_llm(text: str) -> List[str]:
    """Ask LLM to extract named entities (simple JSON list expected)."""
    prompt = f"""
Extract the named entities (people, organizations, locations, products, concepts)
from the following text. Output a JSON array of short strings (entity names) only.

TEXT:
{text}

Output:
"""
    try:
        response = ollama.generate(model=LLM_MODEL, prompt=prompt, format="json")
        # response may contain a field like "response" or "content"
        raw = response.get("response") or response.get("output") or response.get("content") or "[]"
        entities = json.loads(raw)
        if not isinstance(entities, list):
            entities = []
        return [str(e).strip() for e in entities if str(e).strip()]
    except Exception as e:
        print(f"[extract_entities_via_llm] LLM error: {e}")
        return []

# --------------------------
# DB functions: create schema, insert chunks/entities/relations
# --------------------------
def create_schema(cur: Cursor):
    """Create tables for RAG chunks + entities + relations."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Drop if exists for a clean run (comment out in production!)
    cur.execute("DROP TABLE IF EXISTS rag_relations")
    cur.execute("DROP TABLE IF EXISTS rag_entities")
    cur.execute("DROP TABLE IF EXISTS rag_chunks")

    cur.execute(f"""
    CREATE TABLE rag_chunks (
        id SERIAL PRIMARY KEY,
        chunk TEXT NOT NULL,
        embedding VECTOR({EMBED_DIM})
    );
    """)

    cur.execute("""
    CREATE TABLE rag_entities (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE rag_relations (
        chunk_id INT REFERENCES rag_chunks(id),
        entity_id INT REFERENCES rag_entities(id),
        relation_type TEXT DEFAULT 'mentions',
        UNIQUE(chunk_id, entity_id)
    );
    """)

    # optional: index for faster vector search (requires appropriate pgvector index config)
    try:
        cur.execute("CREATE INDEX ON rag_chunks USING ivfflat (embedding) WITH (lists = 100);")
    except Exception as e:
        # index creation might fail depending on pgvector setup — not fatal
        print(f"[create_schema] ivfflat index skipped/failed: {e}")

    print("[create_schema] Schema created")

def save_chunk(cur: Cursor, text: str, embedding: List[float]) -> int:
    if not embedding:
        print("[save_chunk] Empty embedding; skipping insert")
        return -1
    vec_literal = to_pgvector_string(embedding)
    cur.execute(
        "INSERT INTO rag_chunks (chunk, embedding) VALUES (%s, %s::vector) RETURNING id",
        (text, vec_literal)
    )
    cid = cur.fetchone()[0]
    return cid

def get_or_create_entity(cur: Cursor, name: str) -> int:
    cur.execute(
        "INSERT INTO rag_entities (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
        (name,)
    )
    return cur.fetchone()[0]

def link_chunk_entity(cur: Cursor, chunk_id: int, entity_id: int, relation_type: str = "mentions"):
    if chunk_id == -1:
        return
    cur.execute(
        "INSERT INTO rag_relations (chunk_id, entity_id, relation_type) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        (chunk_id, entity_id, relation_type)
    )

# --------------------------
# Neo4j: push entities & relations (optional)
# --------------------------
class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        if not NEO4J_AVAILABLE:
            raise RuntimeError("neo4j package not installed")
        self._drv = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._drv.close()

    def create_entity_node(self, name: str):
        with self._drv.session() as s:
            s.run("MERGE (e:Entity {name: $name}) RETURN e", name=name)

    def create_chunk_node_and_link(self, chunk_id: int, text: str, entities: List[str]):
        with self._drv.session() as s:
            s.run("MERGE (c:Chunk {id: $id}) SET c.text = $text", id=chunk_id, text=text)
            for ent in entities:
                s.run("""
                MERGE (e:Entity {name: $ename})
                MERGE (c:Chunk {id: $cid})
                MERGE (c)-[:MENTIONS]->(e)
                """, ename=ent, cid=chunk_id)

# --------------------------
# Retrieval: Combined Vector + Graph Traversal
# --------------------------
def retrieve_by_vector(cur: Cursor, query: str, k: int = 5) -> List[Dict[str, Any]]:
    emb = create_embedding(query)
    if not emb:
        return []
    vec = to_pgvector_string(emb)
    cur.execute(
        "SELECT id, chunk, embedding <=> %s::vector AS distance FROM rag_chunks ORDER BY distance ASC LIMIT %s",
        (vec, k)
    )
    rows = cur.fetchall()
    return [{"id": r[0], "chunk": r[1], "distance": float(r[2])} for r in rows]

def expand_via_entities(cur: Cursor, chunk_ids: List[int], limit: int = 10) -> List[str]:
    if not chunk_ids:
        return []
    cur.execute(
        """
        SELECT DISTINCT c.chunk
        FROM rag_relations r1
        JOIN rag_relations r2 ON r1.entity_id = r2.entity_id
        JOIN rag_chunks c ON r2.chunk_id = c.id
        WHERE r1.chunk_id = ANY(%s)
        LIMIT %s
        """,
        (chunk_ids, limit)
    )
    return [r[0] for r in cur.fetchall()]

def retrieve_context(cur: Cursor, query: str, k: int = 5) -> List[str]:
    vect_results = retrieve_by_vector(cur, query, k)
    print(f"[retrieve_context] vector results ids: {[r['id'] for r in vect_results]}")
    chunk_ids = [r["id"] for r in vect_results]
    expanded_chunks = expand_via_entities(cur, chunk_ids, limit=20)
    # compose final context: top vector chunks + expanded related chunks
    combined = [r["chunk"] for r in vect_results] + [c for c in expanded_chunks if c not in {r["chunk"] for r in vect_results}]
    print(f"[retrieve_context] combined context size: {len(combined)}")
    return combined

# --------------------------
# Agentic tools (placeholders)
# --------------------------
def get_patient_info(patient_id: str) -> str:
    # placeholder: call a patient DB or service
    return f"Patient info for {patient_id} (placeholder)"

def get_recent_news(topic: str, n: int = 3) -> List[str]:
    # placeholder: call news API
    return [f"News item {i+1} about {topic}" for i in range(n)]

def execute_cypher(neo_client: Any, cypher: str) -> List[Dict[str, Any]]:
    if not NEO4J_AVAILABLE or neo_client is None:
        return [{"error": "Neo4j not configured"}]
    with neo_client._drv.session() as s:
        res = s.run(cypher)
        return [r.data() for r in res]

# --------------------------
# LLM-based agent orchestration (very simple)
# --------------------------
def llm_compose_answer(context_chunks: List[str], user_prompt: str) -> str:
    # Compose a prompt for the LLM with retrieved context
    context_text = "\n\n".join(context_chunks[:10])  # limit size
    prompt = f"""
You are an agent that composes a helpful final answer using the provided context.

CONTEXT:
{context_text}

USER PROMPT:
{user_prompt}

Produce a concise helpful response in French (or in the language of the prompt) using the context. If context is insufficient, state that you need more information.
"""
    try:
        out = ollama.generate(model=LLM_MODEL, prompt=prompt)
        return out.get("response") or out.get("output") or json.dumps(out)
    except Exception as e:
        return f"LLM error: {e}"

# --------------------------
# Main ingestion + retrieval demo
# --------------------------
def main():
    # optionally init neo4j client
    neo_client = None
    if NEO4J_AVAILABLE:
        try:
            neo_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            print("[main] Neo4j client available")
        except Exception as e:
            print(f"[main] Neo4j init failed: {e}")
            neo_client = None

    # connect to Postgres
    with psycopg.connect(DB_CONN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # create schema (comment out if you don't want to drop existing)
            create_schema(cur)

            # ingest files
            files = list(TRANS_FOLDER.glob("*.txt"))
            print(f"[main] found {len(files)} files in {TRANS_FOLDER}")
            for fp in files:
                chunks = read_and_filter(fp)
                for chunk in chunks:
                    emb = create_embedding(chunk)
                    cid = save_chunk(cur, chunk, emb)
                    entities = extract_entities_via_llm(chunk)
                    for ent in entities:
                        eid = get_or_create_entity(cur, ent)
                        link_chunk_entity(cur, cid, eid)
                    # push to neo4j optionally
                    if neo_client:
                        try:
                            neo_client.create_chunk_node_and_link(cid, chunk, entities)
                        except Exception as e:
                            print(f"[main] neo4j push error: {e}")

            # Test retrieval (agentic pipeline)
            user_query = "bonjour je veux de l'aide"
            print("\n--- Retrieval (vector + graph expansion) ---")
            context = retrieve_context(cur, user_query, k=3)
            for i, c in enumerate(context[:10], 1):
                print(f"[context {i}] {c[:200]}")

            print("\n--- Compose final answer with LLM ---")
            final = llm_compose_answer(context, user_query)
            print(final)

    if neo_client:
        neo_client.close()


if __name__ == "__main__":
    main()