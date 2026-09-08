import os
from dotenv import load_dotenv
from src.env_check import ensure_llm_credentials
from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings
from src.vector_store import ChromaVectorStore

load_dotenv()

def main():
    ensure_llm_credentials()

    # 1. Ingestion: Load documents from the data directory
    data_dir = "data"
    print(f"--- Step 1: Loading documents from '{data_dir}' ---")
    documents = load_documents_from_directory(data_dir)
    print(f"Loaded {len(documents)} document pages/objects.\n")

    # 2. Chunking: Split documents into chunks
    print("--- Step 2: Creating chunks ---")
    nodes = create_chunks(documents)
    print(f"Created {len(nodes)} chunks.\n")

    # 3. Chunk Registry: Register chunks
    print("--- Step 3: Registering chunks ---")
    registry = ChunkRegistry()
    registry.register(nodes)
    
    # Save the registry to a file for diagnostic tracking
    os.makedirs("artifacts", exist_ok=True)
    registry_path = "artifacts/chunk_registry.json"
    registry.save_to_json(registry_path)
    print(f"Registry saved to {registry_path}.\n")

    # 4. Embedding: Generate embeddings for each chunk
    print("--- Step 4: Generating embeddings ---")
    embeddings = generate_embeddings(registry)
    print(f"Generated {len(embeddings)} embeddings.\n")

    # 5. Vector Store: Store embeddings in ChromaDB
    print("--- Step 5: Storing in ChromaDB ---")
    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()
    
    vector_store.add_embeddings(embeddings, registry)
    print(f"\nPipeline complete! Total chunks stored in ChromaDB: {vector_store.count()}")

if __name__ == "__main__":
    main()
