# faire les importations nécessaires
import psycopg
from psycopg import Cursor
import ollama
from pathlib import Path  # <-- for handling multiple files

# Déclarer les variables nécessaires
TRANS_FOLDER = Path(r"data\TRANS_TXT")  # <-- dossier contenant tous les .txt
EMBED_MODEL = "embeddinggemma"
db_connection_str = "dbname=rag_chatbot user=postgres password=1803 host=localhost port=5432"

def create_conversation_list(file_path: Path) -> list[str]:
    """Lit un fichier et filtre les lignes inutiles."""
    with open(file_path, "r", encoding="latin-1") as file:
        text_list = file.read().split("\n")

    filtered_list = [
        line.removeprefix("     ")
        for line in text_list
        if not line.startswith("<") and line.strip() != ""
    ]

    print(f"Loaded {file_path.name}: {len(filtered_list)} lines")
    return filtered_list

def calculate_embeddings(corpus: str) -> list[float]:
    response = ollama.embeddings(EMBED_MODEL, corpus)
    return response["embedding"]

def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"

def save_embedding(corpus: str, embedding: list[float], cursor: Cursor) -> None:
    pg_vec = to_pgvector(embedding)
    cursor.execute(
        """
        INSERT INTO embeddings (corpus, embedding)
        VALUES (%s, %s::vector)
        """,
        (corpus, pg_vec),
    )

def similar_corpus(input_corpus: str, k: int, cursor: Cursor):
    embedding = calculate_embeddings(input_corpus)
    pg_vec = to_pgvector(embedding)

    cursor.execute(
        """
        SELECT id, corpus, embedding <=> %s::vector AS distance
        FROM embeddings
        ORDER BY distance ASC
        LIMIT %s
        """,
        (pg_vec, k),
    )

    return cursor.fetchall()


# ------------------------------
# Création + insertion embeddings
# ------------------------------

with psycopg.connect(db_connection_str) as conn:
    conn.autocommit = True

    with conn.cursor() as cur:
        # Supprimer ancienne table
        cur.execute("DROP TABLE IF EXISTS embeddings")

        # Créer extension pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Créer la table embeddings
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                corpus TEXT,
                embedding VECTOR(768)
            );
            """
        )

        # Parcourir tous les fichiers .txt dans TRANS_TXT
        for file_path in TRANS_FOLDER.glob("*.txt"):
            corpus_list = create_conversation_list(file_path)

            # Calcul embeddings + insertion
            for corpus in corpus_list:
                emb = calculate_embeddings(corpus)
                save_embedding(corpus, emb, cur)

        conn.commit()

        # Test similarité
        print("\n--- Résultats similarité (test) ---")
        results = similar_corpus("bonjour je veux de l'aide", 3, cur)
        for r in results:
            print(r)
