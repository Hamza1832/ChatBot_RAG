import json
from pathlib import Path
from typing import List

import psycopg
from psycopg import Cursor

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

# ===========================
# Configuration
# ===========================
TRANS_FOLDER = Path(r"C:\Users\hamza\Desktop\ChatbotRag\ChatBot_RAG\data\TRANS_TXT")
DB_CONN = "dbname=rag_chatbot user=postgres password=1803 host=localhost port=5432"
EMBED_DIM = 768

# ===========================
# LangChain Models
# ===========================
embeddings_model = OllamaEmbeddings(model="embeddinggemma")
llm = ChatOllama(model="llama3", temperature=0)

# ===========================
# Utilities
# ===========================
def read_and_filter(file_path: Path) -> List[str]:
    with open(file_path, "r", encoding="latin-1") as f:
        lines = f.read().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("<")]

def to_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"

# ===========================
# Entity Extraction (LCEL)
# ===========================
entity_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Extract named entities (people, organizations, locations, concepts).
Return ONLY a JSON array of strings.

TEXT:
{text}
"""
)

entity_chain = entity_prompt | llm | StrOutputParser()

def extract_entities(text: str) -> List[str]:
    try:
        res = entity_chain.invoke({"text": text})
        entities = json.loads(res)
        return [e for e in entities if isinstance(e, str)]
    except Exception:
        return []

# ===========================
# Database Schema
# ===========================
def create_schema(cur: Cursor):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    cur.execute("DROP TABLE IF EXISTS doc_relations")
    cur.execute("DROP TABLE IF EXISTS doc_entities")
    cur.execute("DROP TABLE IF EXISTS doc_chunks")

    cur.execute(f"""
    CREATE TABLE doc_chunks (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding VECTOR({EMBED_DIM})
    )
    """)

    cur.execute("""
    CREATE TABLE doc_entities (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE doc_relations (
        chunk_id INT REFERENCES doc_chunks(id),
        entity_id INT REFERENCES doc_entities(id),
        UNIQUE(chunk_id, entity_id)
    )
    """)

# ===========================
# Ingestion
# ===========================
def ingest_documents(cur: Cursor):
    for file in TRANS_FOLDER.glob("*.txt"):
        for chunk in read_and_filter(file):

            embedding = embeddings_model.embed_query(chunk)

            cur.execute(
                """
                INSERT INTO doc_chunks (content, embedding)
                VALUES (%s, %s::vector)
                RETURNING id
                """,
                (chunk, to_pgvector(embedding)),
            )
            chunk_id = cur.fetchone()[0]

            entities = extract_entities(chunk)
            for ent in entities:
                cur.execute(
                    """
                    INSERT INTO doc_entities (name)
                    VALUES (%s)
                    ON CONFLICT (name)
                    DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                    """,
                    (ent,),
                )
                entity_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO doc_relations (chunk_id, entity_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (chunk_id, entity_id),
                )

# ===========================
# Retrieval (Vector + Entity Expansion)
# ===========================
def retrieve_context(cur: Cursor, query: str, k: int = 5) -> List[Document]:
    query_embedding = embeddings_model.embed_query(query)

    cur.execute(
        """
        SELECT id, content
        FROM doc_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (to_pgvector(query_embedding), k),
    )

    base_chunks = cur.fetchall()
    chunk_ids = [r[0] for r in base_chunks]

    if chunk_ids:
        cur.execute(
            """
            SELECT DISTINCT c.content
            FROM doc_relations r1
            JOIN doc_relations r2 ON r1.entity_id = r2.entity_id
            JOIN doc_chunks c ON r2.chunk_id = c.id
            WHERE r1.chunk_id = ANY(%s)
            """,
            (chunk_ids,),
        )
        expanded = [r[0] for r in cur.fetchall()]
    else:
        expanded = []

    texts = [r[1] for r in base_chunks] + expanded
    return [Document(page_content=t) for t in texts]

# ===========================
# Final Answer Chain (LCEL)
# ===========================
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent assistant.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()

def answer_question(cur: Cursor, question: str) -> str:
    docs = retrieve_context(cur, question)
    context_text = "\n\n".join(d.page_content for d in docs[:10])
    return answer_chain.invoke({
        "context": context_text,
        "question": question
    })

# ===========================
# Main
# ===========================
def main():
    with psycopg.connect(DB_CONN) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            create_schema(cur)
            ingest_documents(cur)

            question = "bonjour je veux de l'aide"
            response = answer_question(cur, question)

            print("\n=== Final Answer ===")
            print(response)

if __name__ == "__main__":
    main()
