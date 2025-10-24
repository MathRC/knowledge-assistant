# app/ingest_data.py
import os
import json
import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from urllib.parse import urlparse


# Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
COLLECTION_NAME = "StyleGuide"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHUNKS_FILE_PATH = "data/processed/section_chunks.json"


def get_weaviate_client():
    """Establishes a connection to Weaviate with retry logic for readiness."""
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname
    port = parsed.port or 8080

    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
    )

    # Wait for Weaviate to be ready (required for weaviate-client==4.16.9)
    import time
    for _ in range(60):
        try:
            if client.is_ready():
                print("✅ Connected to Weaviate and service is ready.")
                return client
        except Exception as e:
            print(f"⏳ Waiting for Weaviate... (error: {e})")
            time.sleep(1)

    raise RuntimeError("❌ Failed to connect to Weaviate within 60 seconds.")


def load_chunks():
    """Loads pre-processed document chunks from JSON file."""
    with open(CHUNKS_FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Main execution: connects to Weaviate, creates collection, and imports chunked data with embeddings."""
    print("Starting data ingestion process.")

    client = get_weaviate_client()
    print("Connected to Weaviate.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    if client.collections.exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.collections.delete(COLLECTION_NAME)

    print(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.collections.create(
        name=COLLECTION_NAME,
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="part", data_type=DataType.TEXT),
            Property(name="chapter", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="subsection", data_type=DataType.TEXT),
            Property(name="page_number", data_type=DataType.TEXT),
            Property(name="token_count", data_type=DataType.INT),
            Property(name="method", data_type=DataType.TEXT),
            Property(name="source_document", data_type=DataType.TEXT),
            Property(name="is_split", data_type=DataType.BOOL),
            Property(name="embedding_model", data_type=DataType.TEXT),
        ],
        # Use custom vectors; disable auto-vectorization
        vector_config=[
            {
                "name": "default",
                "vector_index": Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
                "vectorizer": Configure.Vectorizer.none()
            }
        ],
    )

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} section-based chunks")

    print("Generating embeddings and importing into Weaviate...")
    with collection.batch.fixed_size(100) as batch:
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            vector = embedding_model.encode(text).tolist()

            batch.add_object(
                properties={
                    "chunk_id": chunk.get("chunk_id"),
                    "text": text,
                    "part": chunk.get("part", "N/A"),
                    "chapter": chunk.get("chapter", "N/A"),
                    "section": chunk.get("section", "N/A"),
                    "subsection": chunk.get("subsection", "N/A"),
                    "page_number": str(chunk.get("page_number", "N/A")),
                    "token_count": int(chunk.get("token_count", 0)),
                    "method": chunk.get("method", ""),
                    "source_document": chunk.get("source_document", ""),
                    "is_split": bool(chunk.get("is_split", False)),
                    "embedding_model": EMBEDDING_MODEL_NAME,
                },
                vector=vector,
            )

            if (i + 1) % 100 == 0 or (i + 1) == len(chunks):
                print(f"Imported {i + 1}/{len(chunks)} chunks")

    print("Data ingestion completed successfully.")
    client.close()


if __name__ == "__main__":
    main()