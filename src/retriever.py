"""
Retriever Module.
Responsible for converting a user question into an embedding, 
searching the Vector Store (Dense), running BM25 (Sparse), 
and fusing the results using Reciprocal Rank Fusion (RRF).
"""

import time
import logging
import re
from src.cache_utils import serialized_cache
from typing import List, Dict, Any
from dataclasses import dataclass
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from configs.models import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, BGE_QUERY_INSTRUCTION
from configs.pipeline import (
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    FUSION_CANDIDATE_POOL,
    RERANK_INPUT_SIZE,
    RERANKER_MAX_LENGTH,
    RETRIEVAL_MODE,
    RERANK_ENABLED,
    VALID_RETRIEVAL_MODES,
)
from src.vector_store import VectorStore
from src.chunk_registry import ChunkRegistry
from src.device import get_device, describe_device
from src.embedding_engine import get_shared_embed_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """
    Represents a single chunk returned from the Hybrid Search.
    """
    chunk_id: str
    similarity_score: float # The final RRF score
    rank: int
    page_number: str
    source_file: str
    chunk_index: int
    chunk_text: str
    parent_document_id: str = ""
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int = -1
    sparse_rank: int = -1
    rrf_score: float = 0.0
    reranker_score: float = 0.0

@dataclass
class RetrievalResult:
    """
    A comprehensive result object capturing the entire retrieval event.
    Provides complete transparency for future diagnostics.
    """
    question: str
    question_embedding_dimension: int
    retrieved_chunks: List[RetrievedChunk]
    retrieved_chunk_ids: List[str]
    similarity_scores: List[float]
    retrieval_time: float
    top_k: int
    retrieval_metadata: Dict[str, Any]

class Retriever:
    """
    Executes hybrid search (Dense + Sparse) and fuses results via RRF.
    """
    def __init__(self, vector_store: VectorStore, chunk_registry: ChunkRegistry, top_k: int = RETRIEVAL_TOP_K):
        self.vector_store = vector_store
        self.chunk_registry = chunk_registry
        self.top_k = top_k
        device = get_device()
        self.device = device
        self.embed_model = get_shared_embed_model(EMBEDDING_MODEL_NAME)

        logger.info(
            "Loading Cross-Encoder Reranker: %s on %s",
            RERANKER_MODEL_NAME,
            describe_device(),
        )
        self.cross_encoder = CrossEncoder(
            RERANKER_MODEL_NAME, max_length=RERANKER_MAX_LENGTH, device=device
        )
        
        logger.info("Building BM25 Index from ChunkRegistry...")
        self.registry_chunks = list(self.chunk_registry._records.values())
        
        # BM25 Sparse Index Setup
        # We use a regex tokenizer and strip standard English stopwords to prevent
        # common words like 'what', 'does', 'in' from dominating the TF-IDF scores.
        stopwords = {
            "what", "does", "do", "is", "a", "an", "the", "in", "on", "at", "to", "for", 
            "of", "and", "or", "talk", "about", "with", "by", "as", "it", "this", "that",
            "are", "was", "were", "be", "has", "have", "had", "not", "how", "why", "who"
        }
        
        def tokenize(text: str) -> List[str]:
            tokens = re.findall(r'\w+', text.lower())
            return [t for t in tokens if t not in stopwords]
        
        self.bm25_tokenize = tokenize
            
        tokenized_corpus = []
        for record in self.registry_chunks:
            tokenized_corpus.append(self.bm25_tokenize(record.text))
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 Index built with {len(self.registry_chunks)} documents.")

    def rank_candidates(self, question: str, limit: int, mode: str = None) -> tuple:
        """
        Everything up to (but not including) reranking: embed, dense search,
        BM25, RRF fusion, truncate to ``limit``.

        ``mode`` selects which arms run -- ``"vector"``, ``"bm25"`` or
        ``"hybrid"`` (default, from ``configs.pipeline.RETRIEVAL_MODE``). The
        single-arm modes exist so the ablation can measure what fusion actually
        buys; RRF over one arm is just that arm's own ranking, so the fusion
        code path is shared rather than forked.

        Returns ``(chunks_in_rrf_order, fusion_metadata)``. Split out of
        ``retrieve`` so the fusion stage can be measured independently of the
        cross-encoder -- experiments/exp02_reranker_window.py scores one wide
        candidate pool once and then derives every (window, top_n) setting from
        that single pass, instead of re-running retrieval per setting.
        """
        mode = mode or RETRIEVAL_MODE
        if mode not in VALID_RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Valid: {VALID_RETRIEVAL_MODES}")
        use_dense = mode in ("vector", "hybrid")
        use_sparse = mode in ("bm25", "hybrid")

        question_dim = 0
        dense_ranks = {}
        dense_scores = {}

        if use_dense:
            # 1. Embed the user question for Dense Retrieval.
            # BGE models are trained with query/document instruction pairs. Prepending
            # the instruction prefix to queries (not to stored document chunks) improves
            # retrieval precision. See: https://huggingface.co/BAAI/bge-base-en-v1.5
            prefixed_question = BGE_QUERY_INSTRUCTION + question
            question_embedding = self.embed_model.get_text_embedding(prefixed_question)
            question_dim = len(question_embedding)

            # 2a. Dense Search (fetch a wide pool for fusion)
            dense_results = self.vector_store.search(
                query_embedding=question_embedding, top_k=FUSION_CANDIDATE_POOL
            )

            if dense_results and dense_results.get('ids') and len(dense_results['ids'][0]) > 0:
                ids = dense_results['ids'][0]
                distances = dense_results.get('distances', [[0] * len(ids)])[0]
                for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
                    dense_ranks[chunk_id] = rank
                    dense_scores[chunk_id] = distance

        sparse_ranks = {}
        sparse_scores = {}

        if use_sparse:
            # 2b. Sparse Search (BM25)
            tokenized_query = self.bm25_tokenize(question)
            bm25_scores_list = self.bm25.get_scores(tokenized_query)

            # Sort chunks by BM25 score
            sparse_ranking = sorted(
                zip([record.chunk_id for record in self.registry_chunks], bm25_scores_list),
                key=lambda x: x[1], reverse=True
            )

            for rank, (chunk_id, score) in enumerate(sparse_ranking[:FUSION_CANDIDATE_POOL], start=1):
                if score > 0: # Only rank if it actually matched keywords
                    sparse_ranks[chunk_id] = rank
                    sparse_scores[chunk_id] = score

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60 # Standard RRF constant

        all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))

        # Pre-rerank candidate pool signal (CORPUS diagnostic stage): capture the
        # closest dense match across the *entire* candidate pool, before RRF
        # truncates to top_k. Lower distance = more similar (cosine distance).
        pre_rerank_candidate_pool_size = len(all_candidate_ids)
        pre_rerank_min_dense_distance = min(dense_scores.values()) if dense_scores else None

        for chunk_id in all_candidate_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score
            
        # Sort by final RRF score and keep a *wide* slice (``limit``, which
        # callers set to RERANK_INPUT_SIZE), not ``self.top_k``. Truncating to
        # top_k here would make the cross-encoder stage a no-op — it would only
        # reorder the same k chunks RRF already picked, instead of selecting the
        # best k out of a large candidate pool. That is where reranking earns
        # its cost, and experiments/exp02_reranker_window.py quantifies it.
        final_ranking = sorted(
            rrf_scores.items(), key=lambda x: (-x[1], x[0])
        )[:limit]

        retrieved_chunks = []

        for final_rank, (chunk_id, rrf_score) in enumerate(final_ranking, start=1):
            registry_record = self.chunk_registry.get_chunk(chunk_id)
            if not registry_record:
                continue
                
            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                similarity_score=rrf_score,
                rank=final_rank,
                page_number=str(registry_record.page_number),
                source_file=registry_record.source_file,
                chunk_index=registry_record.chunk_index,
                chunk_text=registry_record.text,
                parent_document_id=registry_record.parent_document_id,
                dense_score=dense_scores.get(chunk_id, 0.0),
                sparse_score=sparse_scores.get(chunk_id, 0.0),
                dense_rank=dense_ranks.get(chunk_id, -1),
                sparse_rank=sparse_ranks.get(chunk_id, -1),
                rrf_score=rrf_score
            )
            
            retrieved_chunks.append(retrieved_chunk)

        return retrieved_chunks, {
            "question_embedding_dimension": question_dim,
            "pre_rerank_candidate_pool_size": pre_rerank_candidate_pool_size,
            "pre_rerank_min_dense_distance": pre_rerank_min_dense_distance,
            "retrieval_mode": mode,
        }

    def rerank(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Cross-encoder rerank, in place, returning the chunks sorted by
        reranker score (descending). Does **not** truncate — the caller decides
        the final top_n, which is what makes the window/top_n ratio an explicit
        choice rather than an implicit one buried in this method.
        """
        if not chunks:
            return []

        logger.info(f"Reranking top {len(chunks)} RRF candidates using Cross-Encoder...")

        # Prepare inputs for the cross-encoder: list of [query, chunk_text] pairs
        cross_inp = [[question, chunk.chunk_text] for chunk in chunks]

        # Get cross-encoder scores
        # Larger batches only pay off when the work is parallel; on CPU a big
        # batch just delays the first result without improving throughput.
        rerank_batch = 32 if self.device.startswith("cuda") else 16
        cross_scores = self.cross_encoder.predict(
            cross_inp, batch_size=rerank_batch, show_progress_bar=False
        )

        # Update similarity score and reranker score with cross-encoder score
        for chunk, score in zip(chunks, cross_scores):
            chunk.similarity_score = float(score)
            chunk.reranker_score = float(score)

        ranked = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)
        return ranked

    def retrieve(
        self,
        question: str,
        mode: str = None,
        rerank: bool = None,
        top_n: int = None,
    ) -> RetrievalResult:
        """
        Processes a question and returns the most relevant chunks.

        ``mode`` / ``rerank`` default to ``configs.pipeline``'s
        ``RETRIEVAL_MODE`` / ``RERANK_ENABLED`` so existing callers are
        unchanged; the ablation harness overrides them per arm.
        """
        mode = mode or RETRIEVAL_MODE
        rerank_enabled = RERANK_ENABLED if rerank is None else rerank
        final_n = top_n or RERANKER_TOP_N

        logger.info("Retrieval (mode=%s, rerank=%s) for: '%s'", mode, rerank_enabled, question)
        start_time = time.time()

        # Without a reranker the fusion pool itself is the answer, so there is
        # no reason to score a window wider than the final context size.
        pool = RERANK_INPUT_SIZE if rerank_enabled else final_n
        candidates, fusion_metadata = self.rank_candidates(question, pool, mode=mode)
        question_dim = fusion_metadata["question_embedding_dimension"]

        # 4. Rerank the RRF candidates using Cross-Encoder, then truncate to
        # the final context size.
        if rerank_enabled:
            retrieved_chunks = self.rerank(question, candidates)[:final_n]
        else:
            retrieved_chunks = candidates[:final_n]
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
        similarity_scores = [chunk.similarity_score for chunk in retrieved_chunks]

        # Re-assign final ranks based on cross-encoder sorting
        for i, chunk in enumerate(retrieved_chunks, start=1):
            chunk.rank = i

        retrieval_time = time.time() - start_time

        # 5. Construct RetrievalResult
        result = RetrievalResult(
            question=question,
            question_embedding_dimension=question_dim,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            similarity_scores=similarity_scores,
            retrieval_time=retrieval_time,
            top_k=self.top_k,
            retrieval_metadata={
                "embedding_model": EMBEDDING_MODEL_NAME,
                "bge_query_instruction_applied": mode in ("vector", "hybrid"),
                "vector_store_type": type(self.vector_store).__name__,
                "retrieval_mode": mode,
                "rerank_enabled": rerank_enabled,
                "retrieval_type": f"{mode}{'_cross_encoder' if rerank_enabled else ''}",
                "reranker_model": RERANKER_MODEL_NAME if rerank_enabled else None,
                "rerank_input_size": pool,
                "reranker_top_n": final_n,
                "pre_rerank_candidate_pool_size": fusion_metadata["pre_rerank_candidate_pool_size"],
                "pre_rerank_min_dense_distance": fusion_metadata["pre_rerank_min_dense_distance"],
            }
        )
        
        logger.info(f"Hybrid Retrieval complete in {retrieval_time:.3f}s. Found {len(retrieved_chunks)} chunks.")
        return result


@serialized_cache(maxsize=1)
def get_retriever(vector_store: VectorStore, chunk_registry: ChunkRegistry, top_k: int = RETRIEVAL_TOP_K) -> Retriever:
    """Lazily construct a Retriever (and its heavy models) once per process."""
    return Retriever(vector_store=vector_store, chunk_registry=chunk_registry, top_k=top_k)
