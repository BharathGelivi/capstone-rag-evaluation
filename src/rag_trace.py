"""
RAGTrace Module.

Provides the RAGTrace dataclass and RAGTraceBuilder to construct
and serialize an execution record of a RAG pipeline run.
"""

import os
import json
import uuid
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from src.retriever import RetrievalResult
from src.generator import GenerationResult
from configs.pipeline import CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K, RERANKER_TOP_N
from configs.models import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RAGTrace:
    """
    A canonical, JSON-serializable record of a single RAG execution.
    Does NOT contain evaluations, only facts.
    """
    trace_id: str
    trace_version: str
    pipeline_version: str
    framework_version: str
    timestamp: str
    
    question: str
    generated_answer: str
    
    prompt_snapshot: str
    prompt_length: int
    
    retrieved_chunk_references: List[Dict[str, Any]]
    
    configuration_snapshot: Dict[str, Any]
    execution_statistics: Dict[str, Any]
    pipeline_stage_status: Dict[str, str]
    
    diagnostics: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'RAGTrace':
        data = json.loads(data_str) if isinstance(data_str, str) else data_str
        return cls(**data)

class RAGTraceBuilder:
    """
    Builds a RAGTrace object from pipeline results and saves it to disk.
    """
    
    @staticmethod
    def build(retrieval_result: RetrievalResult, generation_result: GenerationResult, total_pipeline_time: float) -> RAGTrace:
        """
        Validates results and constructs the RAGTrace.
        """
        logger.info("Building RAGTrace...")
        
        # Validation: Check for consistency
        if retrieval_result.retrieved_chunk_ids != generation_result.retrieved_chunk_ids:
            logger.error("Inconsistency detected: Retrieved chunks in RetrievalResult do not match GenerationResult.")
            raise ValueError("Consistency Error: retrieved_chunk_ids mismatch between retrieval and generation results.")
            
        # Construct Retrieved Chunk References (omitting full text)
        chunk_refs = []
        for chunk in retrieval_result.retrieved_chunks:
            chunk_refs.append({
                "chunk_id": chunk.chunk_id,
                "rank": chunk.rank,
                "similarity_score": chunk.similarity_score,
                "rrf_score": getattr(chunk, "rrf_score", 0.0),
                "reranker_score": getattr(chunk, "reranker_score", 0.0),
                "dense_score": chunk.dense_score,
                "sparse_score": chunk.sparse_score,
                "dense_rank": chunk.dense_rank,
                "sparse_rank": chunk.sparse_rank,
                "page_number": chunk.page_number,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "parent_document_id": chunk.parent_document_id
            })
            
        # Create configuration snapshot
        config_snapshot = {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "reranker_model": RERANKER_MODEL_NAME,
            "reranker_top_n": RERANKER_TOP_N,
            "llm_model": generation_result.model_name,
            "llm_temperature": LLM_TEMPERATURE,
            "llm_max_tokens": LLM_MAX_TOKENS
        }
        
        # Create execution statistics
        execution_stats = {
            "retrieval_time": retrieval_result.retrieval_time,
            "generation_time": generation_result.generation_time,
            "total_pipeline_time": total_pipeline_time,
            "pre_rerank_candidate_pool_size": retrieval_result.retrieval_metadata.get("pre_rerank_candidate_pool_size"),
            "pre_rerank_min_dense_distance": retrieval_result.retrieval_metadata.get("pre_rerank_min_dense_distance")
        }
        
        # Status block
        pipeline_stage_status = {
            "document_loader": "completed",
            "chunk_engine": "completed",
            "embedding_engine": "completed",
            "vector_store": "completed",
            "retriever": "completed",
            "generator": "completed"
        }
        
        trace = RAGTrace(
            trace_id=str(uuid.uuid4()),
            trace_version="1.0",
            pipeline_version="1.0",
            framework_version="1.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            question=generation_result.question,
            generated_answer=generation_result.generated_answer,
            prompt_snapshot=generation_result.prompt,
            prompt_length=generation_result.prompt_length,
            retrieved_chunk_references=chunk_refs,
            configuration_snapshot=config_snapshot,
            execution_statistics=execution_stats,
            pipeline_stage_status=pipeline_stage_status,
            diagnostics=None
        )
        
        logger.info(f"RAGTrace built successfully: {trace.trace_id}")
        return trace

    @staticmethod
    def save_to_json(trace: 'RAGTrace', base_dir: str = "artifacts/rag_traces") -> str:
        """
        Saves a RAGTrace to a date-partitioned JSON file.
        Returns the path to the saved file.
        """
        # Create YYYY-MM-DD directory
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_path = os.path.join(base_dir, date_str)
        os.makedirs(dir_path, exist_ok=True)
        
        # Create filename using trace_id
        file_name = f"trace_{trace.trace_id}.json"
        file_path = os.path.join(dir_path, file_name)
        
        logger.info(f"Saving RAGTrace to {file_path}...")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(trace.to_json())
            
        logger.info("RAGTrace saved successfully.")
        return file_path
