"""
Vector Store Module.
Defines an abstraction for vector databases and implements a ChromaDB specific store.
This module strictly handles storage and avoids coupling to embedding logic.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import chromadb

from src.embedding_engine import EmbeddingRecord
from src.chunk_registry import ChunkRegistry
from configs.pipeline import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """
    Abstract interface for a Vector Store.
    This ensures that the RAG diagnostic framework isn't hard-locked into one specific database technology.
    """
    
    @abstractmethod
    def initialize_collection(self) -> None:
        """Initializes or opens the vector store collection."""
        pass
        
    @abstractmethod
    def add_embeddings(self, records: List[EmbeddingRecord], chunk_registry: ChunkRegistry) -> None:
        """Inserts embeddings and their metadata into the store."""
        pass
        
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the vector store for the closest embeddings."""
        pass
        
    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific record by its exact chunk ID."""
        pass
        
    @abstractmethod
    def count(self) -> int:
        """Returns the total number of vectors in the store."""
        pass
        
    @abstractmethod
    def delete_collection(self) -> None:
        """Deletes the collection and all its data."""
        pass


from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction


class _NullEmbeddingFunction(ChromaEmbeddingFunction):
    """Null embedding function to prevent ChromaDB from defaulting to 384-dim all-MiniLM model."""
    def __call__(self, input: List[str]) -> List[List[float]]:
        return [[0.0] * 768 for _ in input]

    def name(self) -> str:
        return "null_embedding_function"


class ChromaVectorStore(VectorStore):
    """
    Implementation of the VectorStore interface using ChromaDB.
    """
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    def initialize_collection(self) -> None:
        """Initializes the ChromaDB client and creates/opens the collection."""
        logger.info(f"Initializing ChromaDB at {self.persist_dir} with collection '{self.collection_name}'")
        
        # Ensure directory exists
        os.makedirs(self.persist_dir, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            
            self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NullEmbeddingFunction(),
                )
            logger.info("ChromaDB collection initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise
            
    def add_embeddings(self, records: List[EmbeddingRecord], chunk_registry: ChunkRegistry) -> None:
        """
        Inserts embeddings into ChromaDB, bringing along text from the registry
        and comprehensive diagnostic metadata from the record.
        """
        if not self.collection:
            raise ValueError("Collection not initialized. Call initialize_collection() first.")
            
        if not records:
            logger.warning("No records provided to add_embeddings.")
            return
            
        logger.info(f"Preparing to insert {len(records)} vectors into ChromaDB...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for record in records:
            # 1. Primary Identifier
            ids.append(record.chunk_id)
            
            # 2. Embedding Vector
            embeddings.append(record.embedding)
            
            # 3. Retrieve Text from Canonical Registry
            chunk_record = chunk_registry.get_chunk(record.chunk_id)
            if not chunk_record:
                logger.warning(f"Chunk ID {record.chunk_id} not found in registry. Using empty text.")
                documents.append("")
                # Use default metadata if missing from registry
                meta = {
                    "parent_document_id": record.parent_document_id,
                    "embedding_model": record.embedding_model,
                    "embedding_dimension": record.embedding_dimension
                }
            else:
                documents.append(chunk_record.text)
                
                # 4. Construct Diagnostic Metadata
                # ChromaDB requires metadata values to be str, int, float, or bool.
                meta = {
                    "parent_document_id": chunk_record.parent_document_id,
                    "source_file": chunk_record.source_file,
                    "page_number": str(chunk_record.page_number), # enforce string
                    "chunk_index": int(chunk_record.chunk_index),
                    "configured_chunk_size": int(chunk_record.configured_chunk_size),
                    "configured_chunk_overlap": int(chunk_record.configured_chunk_overlap),
                    "embedding_model": record.embedding_model,
                    "embedding_dimension": int(record.embedding_dimension)
                }
            
            metadatas.append(meta)
            
        # Chroma rejects a single upsert larger than its own max batch size
        # (5461 at the time of writing). A corpus of a few hundred judgments
        # exceeds that in one call, so the insert is chunked. The batch bound is
        # read from the client where available rather than hardcoded, since it
        # is a property of the installed Chroma build.
        max_batch = getattr(self.client, "get_max_batch_size", lambda: 5000)()
        max_batch = max(1, min(int(max_batch or 5000), 5000))

        try:
            for start in range(0, len(ids), max_batch):
                end = start + max_batch
                self.collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents[start:end],
                )
                logger.info("Upserted %d-%d / %d vectors.", start, min(end, len(ids)), len(ids))
            logger.info(f"Successfully inserted {len(ids)} vectors into collection.")
        except Exception as e:
            logger.error(f"Error inserting vectors into ChromaDB: {e}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Placeholder for searching - to be implemented by Retriever module."""
        # For now, this satisfies the interface but isn't meant to be used deeply yet.
        if not self.collection:
            raise ValueError("Collection not initialized.")
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def publish_registry(self, records, registry, registry_path):
        """Publish a complete new view only after its vectors have been stored."""
        ids = {r.chunk_id for r in records}
        if not ids or ids != set(registry._records):
            raise ValueError("Refusing to publish an empty or incomplete corpus")
        previous = set(self.collection.get(include=[])["ids"])
        overlap = sorted(previous & ids)
        originals = [self.collection.get(ids=overlap[start:start + 5000],
                     include=["embeddings", "documents", "metadatas"])
                     for start in range(0, len(overlap), 5000)]
        try:
            self.add_embeddings(records, registry)
            stored = set(self.collection.get(ids=sorted(ids), include=[])["ids"])
            if stored != ids:
                raise RuntimeError("Vector insertion was incomplete; previous registry was preserved")
            registry.save_to_json(registry_path)
        except Exception:
            # Upserts may have changed reused IDs before a later batch failed.
            for original in originals:
                self.collection.upsert(**{key: original[key] for key in
                    ("ids", "embeddings", "documents", "metadatas") if original.get(key) is not None})
            added = sorted(set(self.collection.get(include=[])["ids"]) & (ids - previous))
            for start in range(0, len(added), 5000):
                self.collection.delete(ids=added[start:start + 5000])
            raise
        obsolete = sorted(previous - ids)
        batch = min(self.client.get_max_batch_size(), 5000)
        for start in range(0, len(obsolete), batch):
            self.collection.delete(ids=obsolete[start:start + batch])

    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific stored record by its exact chunk ID."""
        if not self.collection:
            raise ValueError("Collection not initialized.")
            
        results = self.collection.get(
            ids=[chunk_id],
            include=["embeddings", "metadatas", "documents"]
        )
        
        if not results or not results['ids']:
            return None
            
        # Repackage the output for cleaner consumption
        embeddings = results.get('embeddings')
        return {
            "id": results['ids'][0],
            "embedding": embeddings[0] if embeddings is not None and len(embeddings) > 0 else None,
            "metadata": results['metadatas'][0],
            "document": results['documents'][0]
        }

    def count(self) -> int:
        """Returns the total number of vectors in the collection."""
        if not self.collection:
            raise ValueError("Collection not initialized.")
        return self.collection.count()

    def delete_collection(self) -> None:
        """Deletes the entire collection from the client."""
        if not self.client:
            raise ValueError("Client not initialized.")
        
        logger.warning(f"Deleting collection '{self.collection_name}'!")
        self.client.delete_collection(name=self.collection_name)
        self.collection = None
