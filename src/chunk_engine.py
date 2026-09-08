"""
Chunk Engine Module.

Splits LlamaIndex Document objects into smaller TextNodes (chunks) while
preserving rich provenance metadata required by the diagnostic framework.

Three chunking strategies are supported and are selected via
``configs/pipeline.CHUNKING_STRATEGY``:

  ``"sentence"``      — SentenceSplitter (sentence-boundary aware, fast, default)
  ``"semantic"``      — SemanticSplitterNodeParser (embedding-based breakpoints,
                        higher quality, slower — requires an embed_model argument)
  ``"hierarchical"``  — HierarchicalNodeParser + leaf extraction (multi-granularity
                        context, best for structurally complex documents)

Changing strategy or chunk size requires a full re-ingest.
"""

import enum
import logging
from typing import List, Optional

from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    get_leaf_nodes,
)

from configs.pipeline import CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY

logger = logging.getLogger(__name__)


class ChunkStrategy(str, enum.Enum):
    """Supported chunking strategies.

    Using ``str`` as a mixin makes enum values directly comparable to the
    string literals stored in ``configs/pipeline.py``.
    """
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


def create_chunks(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str = CHUNKING_STRATEGY,
    embed_model=None,  # required when strategy == "semantic"
) -> List[BaseNode]:
    """Split *documents* into chunks using the configured strategy.

    Args:
        documents:    LlamaIndex Document objects produced by the ingestion layer.
        chunk_size:   Maximum chunk size in tokens. Defaults to config value.
        chunk_overlap: Overlap between adjacent chunks in tokens.
        strategy:     One of ``"sentence"``, ``"semantic"``, ``"hierarchical"``.
                      Defaults to the value in ``configs/pipeline.CHUNKING_STRATEGY``.
        embed_model:  A LlamaIndex embedding model instance. Required (and only
                      used) when *strategy* is ``"semantic"``.

    Returns:
        A flat list of TextNodes with injected diagnostic metadata.

    Raises:
        ValueError: If *strategy* is ``"semantic"`` and *embed_model* is None,
                    or if an unknown strategy string is provided.
    """
    try:
        resolved_strategy = ChunkStrategy(strategy)
    except ValueError:
        valid = [s.value for s in ChunkStrategy]
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Valid options: {valid}"
        )

    logger.info(
        "Starting chunking: strategy='%s', chunk_size=%d, chunk_overlap=%d, "
        "documents=%d",
        resolved_strategy.value,
        chunk_size,
        chunk_overlap,
        len(documents),
    )

    raw_nodes = _dispatch(
        resolved_strategy, documents, chunk_size, chunk_overlap, embed_model
    )
    enriched = _inject_metadata(raw_nodes, chunk_size, chunk_overlap)

    logger.info(
        "Chunking complete: strategy='%s', produced %d chunk(s).",
        resolved_strategy.value,
        len(enriched),
    )
    return enriched


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _dispatch(
    strategy: ChunkStrategy,
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    embed_model,
) -> List[BaseNode]:
    """Route to the appropriate LlamaIndex parser based on *strategy*."""
    if strategy is ChunkStrategy.SENTENCE:
        return _sentence_split(documents, chunk_size, chunk_overlap)

    if strategy is ChunkStrategy.SEMANTIC:
        return _semantic_split(documents, chunk_size, embed_model)

    if strategy is ChunkStrategy.HIERARCHICAL:
        return _hierarchical_split(documents, chunk_size, chunk_overlap)

    # Exhaustiveness guard — should never reach here given the enum check above.
    raise ValueError(f"Unhandled strategy: {strategy}")  # pragma: no cover


def _sentence_split(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[BaseNode]:
    """SentenceSplitter: fast, sentence-boundary aware, the safe default."""
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)


def _semantic_split(
    documents: List[Document], chunk_size: int, embed_model
) -> List[BaseNode]:
    """SemanticSplitterNodeParser: splits at embedding-similarity breakpoints.

    Requires a real embed_model instance — raises ValueError if absent.
    buffer_size=1 means the splitter looks one sentence ahead/behind when
    deciding whether to break. breakpoint_percentile_threshold=95 is
    conservative (only breaks at very clear topic shifts).
    """
    if embed_model is None:
        raise ValueError(
            "embed_model must be provided when using strategy='semantic'. "
            "Pass the HuggingFaceEmbedding instance used by the rest of the pipeline."
        )

    # Lazy import — only needed for this strategy so users running the default
    # "sentence" strategy don't pay the import cost.
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
    except ImportError as exc:
        raise ImportError(
            "SemanticSplitterNodeParser is not available. "
            "Ensure llama-index-core >= 0.10 is installed."
        ) from exc

    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )
    logger.info("SemanticSplitter initialised (buffer_size=1, threshold=95th pct).")
    return splitter.get_nodes_from_documents(documents)


def _hierarchical_split(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[BaseNode]:
    """HierarchicalNodeParser: stores context at 3 granularities, returns leaf nodes.

    Chunk sizes [2048, 512, 128] give parent → child → grandchild hierarchy.
    The retriever fetches small leaf chunks (128 tokens) for precision; the
    parent context (512/2048 tokens) is available for the generator via the
    node relationships stored in LlamaIndex's node graph.
    """
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, chunk_size, max(chunk_size // 4, 64)]
    )
    all_nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)
    logger.info(
        "HierarchicalNodeParser: %d total nodes → %d leaf nodes.",
        len(all_nodes),
        len(leaf_nodes),
    )
    return leaf_nodes


def _inject_metadata(
    nodes: List[BaseNode], chunk_size: int, chunk_overlap: int
) -> List[BaseNode]:
    """Attach diagnostic provenance metadata to every TextNode in-place.

    Non-TextNode objects (e.g., ImageNode) are passed through unchanged to
    keep the function safe for future multi-modal pipelines.
    """
    result: List[BaseNode] = []
    for index, node in enumerate(nodes):
        if isinstance(node, TextNode):
            node.metadata["chunk_index"] = index
            node.metadata["chunk_size_config"] = chunk_size
            node.metadata["chunk_overlap_config"] = chunk_overlap
            # start_char_idx / end_char_idx are set by LlamaIndex parsers
            node.metadata["character_start"] = node.start_char_idx
            node.metadata["character_end"] = node.end_char_idx
            node.metadata["parent_document_id"] = node.ref_doc_id

            # Normalise source file / page keys so downstream consumers don't
            # need to know which key a particular reader happened to use.
            if "file_path" in node.metadata and "source_file" not in node.metadata:
                node.metadata["source_file"] = node.metadata["file_path"]
            if "page_label" in node.metadata and "page_number" not in node.metadata:
                node.metadata["page_number"] = node.metadata["page_label"]

        result.append(node)
    return result
