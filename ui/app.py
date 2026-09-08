"""
X-RAG Explainability Platform — Streamlit Application.

A ChatGPT-like interface for the X-RAG framework with:
- Persistent conversation memory
- Session management
- Retrieved chunk inspection
- Claim verification display
- RAG trace timeline
- Memory visualization
- Debug mode
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.memory.memory_manager import MemoryManager
from src.memory.memory_models import MemoryConfig, SessionInfo

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="X-RAG Explainability Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load custom CSS
# ---------------------------------------------------------------------------
# main.css supplies the design tokens and base theme; chat.css layers the
# ChatGPT-style conversation surface on top and must load second.
for _sheet in ("main.css", "chat.css"):
    _path = os.path.join(os.path.dirname(__file__), "styles", _sheet)
    if os.path.exists(_path):
        with open(_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Starter prompts shown on the empty-conversation screen. Chosen to span the
# three acts in the corpus (BNS, BNSS, BSA) so the first answer exercises a
# realistic retrieval path rather than a trivial one.
SUGGESTIONS = [
    "What is the punishment for murder?",
    "What are the rights of an arrested person?",
    "Define culpable homicide not amounting to murder",
    "When is secondary evidence admissible?",
]


def _utc_now_iso() -> str:
    """Current UTC time as a Zulu-suffixed ISO string.

    ``datetime.utcnow()`` is deprecated (it returns a naive datetime that
    silently misrepresents the timezone); this keeps the identical output
    format the trace consumers already expect.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


# ---------------------------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize all session state variables."""
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
        st.session_state.memory_manager.initialize()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = st.session_state.memory_manager.ensure_session()

    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = True

    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None

    if "pipeline_loaded" not in st.session_state:
        st.session_state.pipeline_loaded = False

    if "show_right_panel" not in st.session_state:
        st.session_state.show_right_panel = True

    # Set when a suggestion chip is clicked; consumed by the composer in main().
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Holds the question awaiting an answer across the post-then-answer rerun.
    if "pending_answer" not in st.session_state:
        st.session_state.pending_answer = None

    # Stream tokens as they arrive. Same total latency, ~1s to first token
    # instead of 20-70s of blank spinner.
    if "stream_answers" not in st.session_state:
        st.session_state.stream_answers = True

    # Which corpus to answer from, and which retrieval strategy to use. Both
    # are research dimensions, so they are switchable at runtime rather than
    # baked into config -- the whole point of the study is comparing them.
    if "corpus" not in st.session_state:
        st.session_state.corpus = "statutes"

    if "strategy_arm" not in st.session_state:
        st.session_state.strategy_arm = "C_hybrid_rerank"

    # Metric panels are computed on demand and cached per trace: the RAGAS-family
    # scores cost ~10 LLM judge calls, which must not be paid on every turn.
    if "metric_cache" not in st.session_state:
        st.session_state.metric_cache = {}

    # Claim extraction + NLI verification. Defaults to on when a GPU is
    # present (~3s/claim, worth having by default) and off on CPU, where it
    # costs 30-115s per claim and makes the app feel broken.
    if "deep_analysis" not in st.session_state:
        try:
            from src.device import get_device

            st.session_state.deep_analysis = get_device().startswith("cuda")
        except Exception:
            st.session_state.deep_analysis = False


init_session_state()
mm: MemoryManager = st.session_state.memory_manager


# ---------------------------------------------------------------------------
# Pipeline initialization (heavy models)
# ---------------------------------------------------------------------------
#: corpus key -> (registry path, chroma collection, human label)
CORPORA = {
    "statutes": (
        os.path.join("artifacts", "chunk_registry.json"),
        "rag_benchmark_collection",
        "Statutes (BNS / BNSS / BSA)",
    ),
    "judgments": (
        os.path.join("artifacts", "legal", "chunk_registry_legal.json"),
        "legal_corpus_legal",
        "Supreme Court judgments (legal corpus)",
    ),
}


@st.cache_resource(show_spinner=False)
def load_pipeline(corpus: str = "statutes"):
    """Load the RAG pipeline components for one corpus (cached across reruns).

    Keyed by corpus so switching corpora does not reload the other one's models,
    and so the two registries can coexist in a session.
    """
    from src.env_check import ensure_llm_credentials
    from src.chunk_registry import ChunkRegistry
    from src.vector_store import ChromaVectorStore
    from src.retriever import Retriever
    from src.generator import Generator

    ensure_llm_credentials()

    relative_registry, collection, _label = CORPORA[corpus]
    registry_path = os.path.join(PROJECT_ROOT, relative_registry)
    if not os.path.exists(registry_path):
        hint = ("Run run_pipeline.py first." if corpus == "statutes"
                else "Run `python -m scripts.build_legal_corpus` first.")
        return None, None, None, f"Corpus '{corpus}' is not ingested. {hint}"

    registry = ChunkRegistry.load_from_json(registry_path)
    vector_store = ChromaVectorStore(collection_name=collection)
    vector_store.initialize_collection()
    retriever = Retriever(vector_store, registry)
    generator = Generator()

    return retriever, generator, registry, None


@st.cache_resource(show_spinner=False)
def load_knowledge_graph():
    """Citation graph, loaded once. Absent until the legal corpus is built."""
    from src.legal_graph import GRAPH_PATH, load_graph

    path = os.path.join(PROJECT_ROOT, GRAPH_PATH)
    if not os.path.exists(path):
        return None
    return load_graph(path)


@st.cache_resource(show_spinner=False)
def load_verifier():
    """Load the NLI claim verifier once per process.

    This was previously constructed inside the request handler, so every turn
    loaded a fresh ~1.6 GB deberta-v3-large onto the GPU without releasing the
    previous one. VRAM filled up ("0.0 GB free") and per-claim latency degraded
    6s -> 25s -> 46s across a four-turn conversation as the allocator thrashed.
    """
    from src.claim_verifier import ClaimVerifier

    return ClaimVerifier()


@st.cache_resource(show_spinner=False)
def load_decomposer():
    """Load the claim decomposer once per process (holds an LLM client)."""
    from src.claim_decomposer import ClaimDecomposer

    return ClaimDecomposer()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_rag_pipeline(
    question: str,
    stream: bool = False,
) -> Dict[str, Any]:
    """Execute the full RAG pipeline and return results.

    Args:
        question: The user's question.
        stream:   When True, the answer is rendered token-by-token via
                  ``st.write_stream`` as it arrives. Total time is unchanged —
                  time-to-first-token is not.
    """
    retriever, generator, registry, error = load_pipeline(st.session_state.corpus)

    if error:
        return {"error": error, "answer": error}

    result = {"question": question, "timestamps": {}}

    # Step 1: Memory retrieval
    t0 = time.time()
    memory_results = []
    memory_context = ""
    if st.session_state.memory_enabled:
        try:
            # One search, then format. Calling get_memory_context() as well
            # would repeat the embedding and vector query, and double-count
            # every hit's access_count.
            memory_results = mm.search_memory(
                question, top_k=3, session_id=st.session_state.current_session_id
            )
            memory_context = mm.format_memory_context(
                memory_results, session_id=st.session_state.current_session_id
            )
        except Exception as e:
            memory_results = []
            memory_context = ""
    result["memory_time"] = time.time() - t0
    result["memory_results"] = memory_results
    result["timestamps"]["memory_retrieved"] = _utc_now_iso()

    # Step 1b: Condense follow-ups into a standalone retrieval query.
    # "Can you elaborate on that?" embeds to nothing useful on its own — the
    # referent is in the previous turn, so resolve it before retrieving.
    prior = list(st.session_state.messages)
    # The caller appends the current question to `messages` before invoking the
    # pipeline, so drop that trailing turn — it is passed separately as the
    # final user message and must not also appear in the history.
    if prior and prior[-1].get("role") == "user" and prior[-1].get("content") == question:
        prior = prior[:-1]

    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in prior
        if m.get("role") in ("user", "assistant") and m.get("content")
    ][-8:]

    search_query = question
    if chat_history:
        try:
            search_query = generator.condense_query(question, chat_history)
        except Exception:
            search_query = question
    result["search_query"] = search_query
    result["query_was_condensed"] = search_query != question

    # Step 2: Knowledge retrieval, under the selected strategy.
    # execute_arm is the same function the ablation runs, so what the UI shows
    # and what the experiment measures cannot drift apart.
    t1 = time.time()
    arm = st.session_state.strategy_arm
    from experiments.exp06_strategy_ablation import ARMS, execute_arm
    from src.retriever import RetrievalResult

    if arm == "C_hybrid_rerank":
        retrieval_result = retriever.retrieve(search_query)
    else:
        graph = load_knowledge_graph() if ARMS[arm].get("graph") or ARMS[arm].get(
            "graph_expand") else None
        chunks, strategy_meta, graph_added = execute_arm(
            arm, search_query, registry, retriever, graph, generator.llm)
        retrieval_result = RetrievalResult(
            question=search_query,
            question_embedding_dimension=int(strategy_meta.get("question_embedding_dimension", 0)),
            retrieved_chunks=chunks,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            similarity_scores=[c.similarity_score for c in chunks],
            retrieval_time=time.time() - t1, top_k=len(chunks),
            retrieval_metadata={**strategy_meta, "arm": arm, "graph_added": graph_added},
        )
    result["arm"] = arm
    result["retrieval_time"] = time.time() - t1
    result["retrieval_result"] = retrieval_result
    result["timestamps"]["knowledge_retrieved"] = _utc_now_iso()

    # Step 3: Generation.
    #
    # Two distinct channels feed the model, and conflating them was the
    # original bug:
    #   - chat_history  -> real alternating user/assistant messages, so the
    #                      model can resolve follow-ups natively.
    #   - memory_context -> semantically-recalled turns from *other* sessions
    #                      or far enough back to have fallen out of the window;
    #                      appended to the system prompt as reference material.
    t2 = time.time()
    from configs.prompts import LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS

    system_instructions = LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS
    if memory_context:
        system_instructions += (
            "\n\nRELEVANT PAST CONVERSATIONS (recalled from long-term memory; "
            "cite as [Memory] if you use them):\n"
            + memory_context
        )

    if stream:
        st.write_stream(
            generator.generate_stream(
                retrieval_result,
                system_instructions=system_instructions,
                chat_history=chat_history,
                question_override=question,
            )
        )
        generation_result = generator.last_stream_result
    else:
        generation_result = generator.generate(
            retrieval_result,
            system_instructions=system_instructions,
            chat_history=chat_history,
            question_override=question,
        )
    result["streamed"] = stream
    result["generation_time"] = time.time() - t2
    result["generation_result"] = generation_result
    result["answer"] = generation_result.generated_answer

    # A failed LLM call is not an answer. Surface it as an error and stop:
    # persisting it would poison conversation history and long-term memory with
    # a non-answer that later turns would then try to reason about.
    if generation_result.error:
        result["error"] = generation_result.error
        result["generation_failed"] = True
        result["total_time"] = time.time() - t0
        return result
    result["timestamps"]["generation_complete"] = _utc_now_iso()

    # Step 4: Build RAGTrace
    t3 = time.time()
    from src.rag_trace import RAGTraceBuilder
    total_time = retrieval_result.retrieval_time + generation_result.generation_time
    trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_time)
    trace_path = RAGTraceBuilder.save_to_json(trace)
    result["trace"] = trace
    result["trace_path"] = trace_path
    result["trace_time"] = time.time() - t3
    result["timestamps"]["trace_saved"] = _utc_now_iso()

    # Steps 5-6: Claim decomposition and NLI verification.
    #
    # These are the explainability layer, and on CPU they dominate the turn:
    # nli-deberta-v3-large costs 30-115s *per claim*, so a six-claim answer adds
    # ~6 minutes on top of a ~65s chat response. They are therefore opt-in via
    # the sidebar rather than run on every message.
    if not st.session_state.deep_analysis:
        result["claims"] = None
        result["claim_count"] = 0
        result["claim_time"] = 0.0
        result["verification"] = None
        result["verification_time"] = 0.0
        result["analysis_skipped"] = True
        result["timestamps"]["claims_extracted"] = _utc_now_iso()
        result["timestamps"]["verification_complete"] = _utc_now_iso()
        _save_to_memory(result, question, generation_result, trace)
        result["total_time"] = time.time() - t0
        return result

    # Step 5: Claim decomposition
    t4 = time.time()
    try:
        decomposer = load_decomposer()
        candidate_claim_set = decomposer.decompose(trace)
        result["claims"] = candidate_claim_set
        result["claim_count"] = candidate_claim_set.total_candidates
    except Exception as e:
        result["claims"] = None
        result["claim_count"] = 0
        result["claim_error"] = str(e)
    result["claim_time"] = time.time() - t4
    result["timestamps"]["claims_extracted"] = _utc_now_iso()

    # Step 6: Claim verification
    t5 = time.time()
    try:
        if result.get("claims"):
            verifier = load_verifier()
            verification = verifier.verify_all(
                result["claims"], trace.trace_id, retrieval_result.retrieved_chunks
            )
            result["verification"] = verification
        else:
            result["verification"] = None
    except Exception as e:
        result["verification"] = None
        result["verification_error"] = str(e)
    result["verification_time"] = time.time() - t5
    result["timestamps"]["verification_complete"] = _utc_now_iso()

    # Step 7: Save to memory
    _save_to_memory(result, question, generation_result, trace)

    result["total_time"] = time.time() - t0
    return result


def _save_to_memory(result, question, generation_result, trace) -> None:
    """Persist the interaction to long-term memory.

    Shared by both the fast path and the deep-analysis path so a turn is
    remembered identically either way — memory must not depend on whether the
    explainability layer ran.
    """
    t6 = time.time()
    if st.session_state.memory_enabled:
        try:
            claim_ids = []
            if result.get("claims"):
                claim_ids = [c.candidate_id for c in result["claims"].candidate_claims]

            mm.save_interaction(
                question=question,
                answer=generation_result.generated_answer,
                session_id=st.session_state.current_session_id,
                trace_id=trace.trace_id,
                retrieved_chunk_ids=result["retrieval_result"].retrieved_chunk_ids,
                claim_ids=claim_ids,
            )
        except Exception:
            pass
    result["memory_save_time"] = time.time() - t6
    result["timestamps"]["saved_to_memory"] = _utc_now_iso()


def format_time(seconds: float) -> str:
    """Format seconds as a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def render_badges(meta: Dict[str, Any]) -> None:
    """Render the metric pill row shown under an assistant answer."""
    pills = []
    if meta.get("retrieval_time"):
        pills.append(("🔍", format_time(meta["retrieval_time"]), False))
    if meta.get("generation_time"):
        pills.append(("⚡", format_time(meta["generation_time"]), False))
    if meta.get("chunk_count"):
        pills.append(("📄", f"{meta['chunk_count']} chunks", False))
    if meta.get("claim_count"):
        pills.append(("📋", f"{meta['claim_count']} claims", False))
    if meta.get("memory_count"):
        pills.append(("🧠", f"{meta['memory_count']} recalled", True))
    if meta.get("query_was_condensed"):
        pills.append(("↻", "follow-up resolved", True))
    if meta.get("total_time"):
        pills.append(("⏱", f"{format_time(meta['total_time'])} total", False))

    if not pills:
        return

    html = "".join(
        f'<span class="xrag-badge{" is-accent" if accent else ""}">{icon} {label}</span>'
        for icon, label, accent in pills
    )
    st.markdown(f'<div class="xrag-badges">{html}</div>', unsafe_allow_html=True)


def get_confidence_color(score: float) -> str:
    """Get a color based on a confidence score."""
    if score >= 0.7:
        return "#10b981"  # green
    elif score >= 0.4:
        return "#f59e0b"  # amber
    return "#ef4444"  # red


def render_session_title_from_question(question: str) -> str:
    """Generate a short session title from the first question."""
    words = question.split()[:6]
    title = " ".join(words)
    if len(question.split()) > 6:
        title += "..."
    return title


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the left sidebar with session management and settings."""
    with st.sidebar:
        # Logo and title
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="font-size: 1.5rem; margin: 0; color: #7c3aed;">🔬 X-RAG</h1>
                <p style="font-size: 0.75rem; color: #9ca3af; margin: 0;">Explainability Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # New session button
        if st.button("➕ New Session", use_container_width=True, type="primary"):
            session = mm.create_session("New Session")
            st.session_state.current_session_id = session.session_id
            st.session_state.messages = []
            st.session_state.last_pipeline_result = None
            st.rerun()

        st.divider()

        # Session list
        st.markdown("### 💬 Sessions")
        sessions = mm.list_sessions()

        for session in sessions:
            is_active = session.session_id == st.session_state.current_session_id
            col1, col2 = st.columns([4, 1])

            with col1:
                btn_type = "primary" if is_active else "secondary"
                label = f"{'▶ ' if is_active else ''}{session.title}"
                if st.button(
                    label,
                    key=f"session_{session.session_id}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    if not is_active:
                        # Load session messages from memory
                        st.session_state.current_session_id = session.session_id
                        mm.switch_session(session.session_id)
                        memories = mm.get_session_memories(session.session_id)
                        st.session_state.messages = []
                        for mem in memories:
                            st.session_state.messages.append(
                                {"role": "user", "content": mem.question}
                            )
                            st.session_state.messages.append(
                                {"role": "assistant", "content": mem.answer}
                            )
                        st.session_state.last_pipeline_result = None
                        st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{session.session_id}", help="Delete session"):
                    mm.delete_session(session.session_id)
                    if is_active:
                        st.session_state.messages = []
                        st.session_state.last_pipeline_result = None
                        new_sessions = mm.list_sessions()
                        if new_sessions:
                            st.session_state.current_session_id = new_sessions[0].session_id
                        else:
                            new_s = mm.create_session("New Session")
                            st.session_state.current_session_id = new_s.session_id
                    st.rerun()

        st.divider()

        # Settings
        st.markdown("### ⚙️ Settings")

        st.session_state.corpus = st.selectbox(
            "📚 Corpus",
            list(CORPORA),
            index=list(CORPORA).index(st.session_state.corpus),
            format_func=lambda key: CORPORA[key][2],
            help="Which ingested corpus answers are retrieved from.",
        )

        from experiments.exp06_strategy_ablation import ARMS as _ARMS

        st.session_state.strategy_arm = st.selectbox(
            "🧭 Retrieval strategy",
            list(_ARMS),
            index=list(_ARMS).index(st.session_state.strategy_arm),
            help=(
                "The same arms the ablation measures. C_hybrid_rerank is the shipped "
                "pipeline; IRCoT and the agentic controller cost extra LLM calls per turn."
            ),
        )

        st.session_state.memory_enabled = st.toggle(
            "🧠 Memory",
            value=st.session_state.memory_enabled,
            help="Enable/disable conversation memory",
        )

        st.session_state.stream_answers = st.toggle(
            "⚡ Stream Answers",
            value=st.session_state.stream_answers,
            help="Render tokens as they arrive instead of waiting for the "
                 "whole answer. Same total time, far less waiting.",
        )

        st.session_state.deep_analysis = st.toggle(
            "🔬 Deep Analysis",
            value=st.session_state.deep_analysis,
            help=(
                "Extract atomic claims and verify each against the retrieved "
                "chunks with an NLI model. Powerful, but it runs on CPU and "
                "adds roughly a minute per claim — leave it off for normal chat."
            ),
        )

        st.session_state.debug_mode = st.toggle(
            "🐛 Debug Mode",
            value=st.session_state.debug_mode,
            help="Show detailed pipeline information",
        )

        st.session_state.show_right_panel = st.toggle(
            "📊 Details Panel",
            value=st.session_state.show_right_panel,
            help="Show the right-side details panel",
        )

        st.divider()

        # Compute device — makes an accidental CPU-only torch install obvious
        # instead of silently costing minutes per turn.
        try:
            from src.device import get_device, describe_device

            on_gpu = get_device().startswith(("cuda", "mps"))
            st.markdown(
                f"### {'🚀' if on_gpu else '🐢'} Device\n"
                f"<span style='font-size:0.75rem;color:#9ca3af;'>{describe_device()}</span>",
                unsafe_allow_html=True,
            )
            if not on_gpu:
                st.caption("Running on CPU — see docs/gpu_setup.md")
        except Exception:
            pass

        st.divider()

        # Memory stats
        st.markdown("### 📊 Memory Stats")
        try:
            stats = mm.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memories", stats["total_memories"])
            with col2:
                st.metric("Sessions", stats["total_sessions"])
        except Exception:
            st.info("Memory not initialized")

        st.divider()

        # Export / Clear
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export", use_container_width=True, help="Export current session"):
                try:
                    export_data = mm.export_session(
                        st.session_state.current_session_id, format="json"
                    )
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(export_data, indent=2, default=str),
                        file_name=f"session_export_{st.session_state.current_session_id}.json",
                        mime="application/json",
                    )
                except Exception as e:
                    st.error(f"Export failed: {e}")
        with col2:
            if st.button("🧹 Clear", use_container_width=True, help="Clear current session memories"):
                mm.clear_session_memory(st.session_state.current_session_id)
                st.session_state.messages = []
                st.session_state.last_pipeline_result = None
                st.rerun()


# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------
def render_chat():
    """Render the main chat interface."""
    # Header
    current_session = mm.get_current_session()
    session_title = current_session.title if current_session else "X-RAG Chat"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0; margin-bottom: 0.5rem;">
            <h2 style="margin: 0; color: #e5e7eb;">💬 {session_title}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="xrag-hero">
                    <h1>What would you like to know?</h1>
                    <p>Ask about the statutes in your corpus — every answer is
                       retrieved, traced, and claim-verified.</p>
                </div>
                <div class="xrag-suggest-label">Try one of these</div>
                """,
                unsafe_allow_html=True,
            )

            # Suggestion chips. Clicking one queues it through the same path as
            # the composer, so there is a single code path for asking.
            for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
                cols = st.columns(len(row))
                for col, suggestion in zip(cols, row):
                    with col:
                        if st.button(suggestion, use_container_width=True,
                                     key=f"suggest_{suggestion[:20]}"):
                            st.session_state.pending_prompt = suggestion
                            st.rerun()
        else:
            for msg in st.session_state.messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🔬"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("metadata"):
                        render_badges(msg["metadata"])

    # A question is answered on the rerun *after* it is posted, so the user's
    # message paints immediately and the thinking indicator sits in the right
    # place. `pending_answer` carries the question across that rerun.
    prompt = st.session_state.pop("pending_answer", None)

    if prompt:
        # Update session title if it's the first message
        if current_session and current_session.title == "New Session":
            new_title = render_session_title_from_question(prompt)
            mm.rename_session(st.session_state.current_session_id, new_title)

        # Generate response
        with st.chat_message("assistant", avatar="🔬"):
            streaming = st.session_state.stream_answers

            # The thinking indicator is only used for the blocking path.
            # Mixing an st.empty() placeholder with write_stream in the same
            # container proved unreliable: the placeholder's clear is not
            # flushed to the browser before the stream begins, so the
            # indicator stayed glued to the answer. Streaming text is its own
            # progress signal, so simply omit the indicator there.
            status = None
            if not streaming:
                status = st.empty()
                status.markdown(
                    '<div class="xrag-thinking"><span class="dot"></span>'
                    '<span class="dot"></span><span class="dot"></span>'
                    "Retrieving and reasoning…</div>",
                    unsafe_allow_html=True,
                )

            result = run_rag_pipeline(prompt, stream=streaming)

            if status is not None:
                status.empty()

            if result.get("generation_failed"):
                # Drop the user's turn back into the composer rather than
                # leaving a dangling question with no reply, and keep the
                # failure out of `messages` so history stays clean.
                st.error(f"⚠️ {result['answer']}")
                st.caption(f"Details: `{result['error']}`")
                if st.session_state.messages and \
                        st.session_state.messages[-1].get("role") == "user":
                    st.session_state.messages.pop()
                if st.button("🔄 Retry", key="retry_failed"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
            elif "error" in result and result.get("answer") == result.get("error"):
                st.error(result["answer"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )
            else:
                # A streamed answer has already been painted by write_stream;
                # re-rendering it here would show the text twice.
                if not result.get("streamed"):
                    st.markdown(result["answer"])

                rr = result.get("retrieval_result")
                meta = {
                    "retrieval_time": result.get("retrieval_time", 0),
                    "generation_time": result.get("generation_time", 0),
                    "claim_count": result.get("claim_count", 0),
                    "memory_count": len(result.get("memory_results", [])),
                    "chunk_count": len(rr.retrieved_chunks) if rr else 0,
                    "query_was_condensed": result.get("query_was_condensed", False),
                    "total_time": result.get("total_time", 0),
                }
                render_badges(meta)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": meta,
                    }
                )

                st.session_state.last_pipeline_result = result


# ---------------------------------------------------------------------------
# RIGHT PANEL (Details)
# ---------------------------------------------------------------------------
def render_right_panel():
    """Render the right-side details panel with tabs."""
    result = st.session_state.last_pipeline_result

    if not result:
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 1rem; color: #6b7280;">
                <p>💡 Ask a question to see pipeline details here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["📄 Chunks", "🧠 Memory", "✅ Claims", "📊 Trace", "🔍 Search",
         "📐 Metrics", "🧭 Strategy"]
    )

    # Tab 1: Retrieved Chunks
    with tab1:
        rr = result.get("retrieval_result")
        if rr and rr.retrieved_chunks:
            st.markdown(f"**{len(rr.retrieved_chunks)} chunks retrieved** in {format_time(result.get('retrieval_time', 0))}")
            for chunk in rr.retrieved_chunks:
                score_color = get_confidence_color(chunk.similarity_score)
                with st.expander(
                    f"Chunk {chunk.rank} — {chunk.chunk_id[:20]}... (Score: {chunk.similarity_score:.3f})",
                    expanded=chunk.rank == 1,
                ):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Source:** {os.path.basename(chunk.source_file)}")
                    with col2:
                        st.markdown(f"**Page:** {chunk.page_number}")
                    with col3:
                        st.markdown(f"**Reranker:** {chunk.reranker_score:.3f}")

                    st.markdown("---")
                    st.markdown(chunk.chunk_text[:500] + ("..." if len(chunk.chunk_text) > 500 else ""))

                    if st.session_state.debug_mode:
                        st.json({
                            "dense_score": chunk.dense_score,
                            "sparse_score": chunk.sparse_score,
                            "rrf_score": chunk.rrf_score,
                            "dense_rank": chunk.dense_rank,
                            "sparse_rank": chunk.sparse_rank,
                        })
        else:
            st.info("No chunks retrieved.")

    # Tab 2: Memory
    with tab2:
        mem_results = result.get("memory_results", [])
        if mem_results:
            st.markdown(f"**{len(mem_results)} relevant memories** found in {format_time(result.get('memory_time', 0))}")
            for i, mr in enumerate(mem_results, 1):
                with st.expander(
                    f"Memory {i} — Score: {mr.final_score:.3f}",
                    expanded=i == 1,
                ):
                    st.markdown(f"**Q:** {mr.memory.question}")
                    st.markdown(f"**A:** {mr.memory.answer[:300]}...")
                    st.markdown(f"**Session:** {mr.memory.session_id}")
                    st.markdown(f"**Time:** {mr.memory.timestamp}")
                    st.markdown(f"**Why:** {mr.retrieval_reason}")

                    if st.session_state.debug_mode:
                        st.json({
                            "semantic_score": mr.semantic_score,
                            "recency_score": mr.recency_score,
                            "frequency_score": mr.frequency_score,
                            "importance_score": mr.importance_score,
                            "final_score": mr.final_score,
                        })
        else:
            st.info("No relevant memories found." if st.session_state.memory_enabled else "Memory is disabled.")

    # Tab 3: Claims
    with tab3:
        claims = result.get("claims")
        verification = result.get("verification")

        if result.get("analysis_skipped"):
            st.info(
                "Claim verification is off. Enable **🔬 Deep Analysis** in the "
                "sidebar to extract atomic claims and verify each against the "
                "retrieved chunks.\n\n"
                "It is disabled by default because the NLI model runs on CPU "
                "and adds roughly a minute per claim."
            )
        elif claims and claims.candidate_claims:
            st.markdown(f"**{claims.total_candidates} claims** extracted in {format_time(result.get('claim_time', 0))}")

            if verification:
                cols = st.columns(4)
                with cols[0]:
                    st.metric("✅ Supported", verification.supported_claims)
                with cols[1]:
                    st.metric("⚠️ Partial", verification.partially_supported_claims)
                with cols[2]:
                    st.metric("❌ Contradicted", verification.contradicted_claims)
                with cols[3]:
                    st.metric("❓ Unsupported", verification.unsupported_claims)

                for vr in verification.results:
                    status_icons = {
                        "SUPPORTED": "✅",
                        "PARTIALLY_SUPPORTED": "⚠️",
                        "CONTRADICTED": "❌",
                        "UNSUPPORTED": "❓",
                        "NOT_VERIFIABLE": "🔍",
                    }
                    icon = status_icons.get(vr.verification_status.value, "❓")
                    with st.expander(
                        f"{icon} {vr.claim_text[:80]}...",
                        expanded=False,
                    ):
                        st.markdown(f"**Status:** {vr.verification_status.value}")
                        st.markdown(f"**Confidence:** {vr.confidence:.3f}")
                        st.markdown(f"**Reason:** {vr.verification_reason}")
                        if vr.evidence_text:
                            st.markdown(f"**Evidence:** {vr.evidence_text[:200]}...")
                        if vr.best_chunk_id:
                            st.markdown(f"**Source Chunk:** {vr.best_chunk_id}")
            else:
                for c in claims.candidate_claims:
                    st.markdown(f"- {c.claim_text}")
        else:
            st.info("No claims extracted." if not result.get("claim_error") else f"Error: {result.get('claim_error')}")

    # Tab 4: Trace
    with tab4:
        trace = result.get("trace")
        if trace:
            st.markdown(f"**Trace ID:** `{trace.trace_id}`")
            st.markdown(f"**Model:** {result.get('generation_result', {}).model_name if hasattr(result.get('generation_result'), 'model_name') else 'N/A'}")

            # Timeline
            st.markdown("### Pipeline Timeline")
            timestamps = result.get("timestamps", {})
            timeline_items = [
                ("🧠 Memory Retrieved", result.get("memory_time", 0)),
                ("🔍 Knowledge Retrieved", result.get("retrieval_time", 0)),
                ("⚡ Answer Generated", result.get("generation_time", 0)),
                ("📋 Claims Extracted", result.get("claim_time", 0)),
                ("✅ Claims Verified", result.get("verification_time", 0)),
                ("💾 Saved to Memory", result.get("memory_save_time", 0)),
            ]

            for step_name, step_time in timeline_items:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{step_name}")
                with col2:
                    st.markdown(f"`{format_time(step_time)}`")

            st.markdown(f"**Total: {format_time(result.get('total_time', 0))}**")

            if st.session_state.debug_mode:
                st.markdown("### Full Trace")
                try:
                    st.json(json.loads(trace.to_json()))
                except Exception:
                    st.code(str(trace))

                st.markdown("### Prompt Snapshot")
                gen_result = result.get("generation_result")
                if gen_result:
                    st.code(gen_result.prompt[:2000], language="text")
        else:
            st.info("No trace available.")

    # Tab 5: Memory Search
    with tab5:
        st.markdown("### 🔍 Search Memories")
        search_query = st.text_input("Search questions, answers, claims...", key="memory_search_input")
        if search_query:
            search_results = mm.search_memory(search_query, top_k=10)
            if search_results:
                for i, sr in enumerate(search_results, 1):
                    with st.expander(f"{i}. {sr.memory.question[:60]}... (Score: {sr.final_score:.3f})"):
                        st.markdown(f"**Q:** {sr.memory.question}")
                        st.markdown(f"**A:** {sr.memory.answer[:300]}...")
                        st.markdown(f"**Session:** {sr.memory.session_id}")
                        st.markdown(f"**Timestamp:** {sr.memory.timestamp}")
                        if sr.memory.trace_id:
                            st.markdown(f"**Trace:** {sr.memory.trace_id}")
            else:
                st.info("No matching memories found.")

        # Memory statistics
        st.markdown("### 📊 Memory Statistics")
        try:
            stats = mm.get_statistics()
            st.json(stats)
        except Exception:
            st.info("Statistics unavailable.")


    # Tab 6: Evaluation metrics
    with tab6:
        render_metrics_tab(result)

    # Tab 7: Strategy trace
    with tab7:
        rr = result.get("retrieval_result")
        if rr is not None:
            from ui.components.metrics import render_strategy_trace

            st.markdown(f"**Arm:** `{result.get('arm', 'C_hybrid_rerank')}`")
            render_strategy_trace(rr.retrieval_metadata)
        else:
            st.info("No retrieval metadata for this turn.")


def render_metrics_tab(result):
    """Compute and show the evaluation panel for the last turn, on request."""
    from ui.components.metrics import render_citation_report, render_metric_panel

    trace = result.get("trace")
    if trace is None:
        st.info("Ask a question first.")
        return

    cache_key = trace.trace_id
    cached = st.session_state.metric_cache.get(cache_key)

    st.markdown(
        "Thirteen scores across the RAGAS, RAGChecker and ARES metric families, plus "
        "citation correctness. Computed on demand: the judged metrics cost roughly ten "
        "LLM calls, so they are not run on every turn."
    )

    reference = st.text_area(
        "Reference answer (optional)",
        value=(cached or {}).get("reference", ""),
        help="Context recall, answer correctness and claim recall need a reference. "
             "Without one they stay 'not computed' rather than being scored 0 or 1.",
        height=80,
    )

    if st.button("Compute metrics", type="primary", key=f"metrics_{cache_key}"):
        with st.spinner("Scoring answer, evidence and citations..."):
            cached = compute_metrics_for_result(result, reference)
            st.session_state.metric_cache[cache_key] = cached

    if not cached:
        st.caption("Not computed yet for this answer.")
        return

    render_metric_panel(
        cached["panel"],
        has_reference=bool(cached.get("reference")),
        has_verification=cached.get("has_verification", False),
        cited_anything=cached.get("cited_anything", False),
    )
    st.markdown("---")
    st.markdown("##### Citation validation")
    render_citation_report(cached.get("citation_report"))


def compute_metrics_for_result(result, reference: str):
    """Assemble the metric panel from the components already loaded in session.

    Reuses ``RagasEvaluator``, ``ClaimVerifier`` and ``AnswerCorrectnessEvaluator``
    rather than reimplementing any metric here.
    """
    from src.citation_check import load_corpus_index, validate_answer_citations
    from src.ragas_metrics import RagasEvaluator
    from src.rag_eval import compute_metric_panel

    retriever, generator, registry, error = load_pipeline(st.session_state.corpus)
    if error:
        st.error(error)
        return None

    chunks = result["retrieval_result"].retrieved_chunks
    answer = result["answer"]
    question = result["question"]
    verification = result.get("verification")

    verifier = load_verifier()
    ragas = RagasEvaluator(llm=generator.llm, embed_model=retriever.embed_model,
                           claim_verifier=verifier)

    answer_correctness = None
    if reference.strip():
        from src.answer_correctness_evaluator import AnswerCorrectnessEvaluator

        answer_correctness = AnswerCorrectnessEvaluator(
            decomposer=load_decomposer(), verifier=verifier
        ).evaluate(answer, reference.strip(), result["trace"].trace_id)

    citation_report = validate_answer_citations(
        answer, chunks, registry, load_corpus_index())

    panel = compute_metric_panel(
        question=question, answer=answer, retrieved_chunks=chunks,
        verification=verification, reference=reference.strip() or None,
        ragas_evaluator=ragas, answer_correctness=answer_correctness,
        citation_report=citation_report,
    )

    return {
        "panel": panel,
        "citation_report": citation_report,
        "reference": reference.strip(),
        "has_verification": verification is not None,
        "cited_anything": citation_report.total_citations > 0,
    }


# ---------------------------------------------------------------------------
# BOTTOM PANEL (Timeline for last query)
# ---------------------------------------------------------------------------
def render_bottom_panel():
    """Render the bottom timeline panel."""
    result = st.session_state.last_pipeline_result
    if not result:
        return

    st.markdown("---")
    st.markdown("### ⏱ Pipeline Timeline")

    steps = [
        ("Question", "📝", 0),
        ("Memory", "🧠", result.get("memory_time", 0)),
        ("Retrieval", "🔍", result.get("retrieval_time", 0)),
        ("Generation", "⚡", result.get("generation_time", 0)),
        ("Claims", "📋", result.get("claim_time", 0)),
        ("Verification", "✅", result.get("verification_time", 0)),
        ("Memory Save", "💾", result.get("memory_save_time", 0)),
    ]

    cols = st.columns(len(steps))
    for i, (name, icon, t) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.5rem; background: rgba(124, 58, 237, 0.1); border-radius: 0.5rem; border: 1px solid rgba(124, 58, 237, 0.2);">
                    <div style="font-size: 1.2rem;">{icon}</div>
                    <div style="font-size: 0.7rem; color: #c4b5fd;">{name}</div>
                    <div style="font-size: 0.65rem; color: #9ca3af;">{format_time(t)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown("")


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
def main():
    render_sidebar()

    if st.session_state.show_right_panel:
        # Two-column layout: chat + details panel
        chat_col, detail_col = st.columns([3, 2])

        with chat_col:
            render_chat()

        with detail_col:
            render_right_panel()
    else:
        # Full-width chat
        render_chat()

    # Bottom timeline
    render_bottom_panel()

    # The composer is created at top level, not inside a column — that is what
    # makes Streamlit dock it to the bottom of the viewport instead of laying
    # it out inline above the transcript.
    typed = st.chat_input("Ask a question about your documents…")
    prompt = typed or st.session_state.pop("pending_prompt", None)

    if prompt:
        # Post the user's turn and rerun immediately, so it paints without
        # waiting on the (slow) pipeline. The answer is produced next run.
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_answer = prompt
        st.rerun()


if __name__ == "__main__":
    main()
