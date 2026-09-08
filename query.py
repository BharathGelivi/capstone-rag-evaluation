import os
import sys
import argparse
from dotenv import load_dotenv

# Windows' default console codepage (cp1252) can't encode characters models
# routinely emit (em dashes, smart quotes, narrow no-break spaces), which
# crashes any print() of generated text before the trace is saved. Force
# UTF-8 stdout/stderr so this script never dies on its own output.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from src.env_check import ensure_llm_credentials
from src.chunk_registry import ChunkRegistry
from src.vector_store import ChromaVectorStore
from src.retriever import get_retriever
from src.generator import Generator

# Load environment variables from .env file
load_dotenv()

def main():
    ensure_llm_credentials()

    parser = argparse.ArgumentParser(description="Query the RAG benchmark system.")
    parser.add_argument("query", type=str, help="The question to ask the system.")
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Optional ground-truth reference answer. Enables reference-based RAGAS "
             "metrics (Context Recall, Answer Similarity, Answer Correctness)."
    )
    args = parser.parse_args()

    # 1. Initialize the Vector Store
    print("--- Initializing Vector Store ---")
    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()

    # 2. Load the Chunk Registry
    print("--- Loading Chunk Registry ---")
    registry_path = "artifacts/chunk_registry.json"
    if not os.path.exists(registry_path):
        print(f"Error: {registry_path} not found. Please run run_pipeline.py first.")
        return
    registry = ChunkRegistry.load_from_json(registry_path)

    # 3. Retrieve chunks
    print(f"\n--- Retrieving context for: '{args.query}' ---")
    retriever = get_retriever(vector_store, registry)
    retrieval_result = retriever.retrieve(args.query)

    print(f"Found {len(retrieval_result.retrieved_chunks)} relevant chunks in {retrieval_result.retrieval_time:.2f}s.")
    for i, chunk in enumerate(retrieval_result.retrieved_chunks, start=1):
        print(f"Retrieved Chunk {i}")
        print("-" * 40)
        print(f"\nScore:\n{chunk.similarity_score:.3f}")
        print(f"\nSource:\n{chunk.source_file}")
        print(f"\nPage:\n{chunk.page_number}")
        print(f"\nChunk ID:\n{chunk.chunk_id}")
        print(f"\nText:\n\n{chunk.chunk_text}")
        print("\n" + "-" * 40 + "\n")

    # 4. Generate Answer
    print("\n--- Generating Answer ---")
    generator = Generator()
    generation_result = generator.generate(retrieval_result)

    print(f"\nAnswer generated in {generation_result.generation_time:.2f}s:")
    print("=" * 60)
    print(generation_result.generated_answer)
    print("=" * 60)

    # 5. Build RAGTrace
    print("\n--- Building RAGTrace ---")
    from src.rag_trace import RAGTraceBuilder
    total_time = retrieval_result.retrieval_time + generation_result.generation_time
    trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_time)
    trace_path = RAGTraceBuilder.save_to_json(trace)
    print(f"RAGTrace saved to {trace_path}")

    # 6. Decompose Claims
    print("\n--- Decomposing Answer into Claims ---")
    from src.claim_decomposer import ClaimDecomposer
    decomposer = ClaimDecomposer()
    candidate_claim_set = decomposer.decompose(trace)
    print(f"Extracted {candidate_claim_set.total_candidates} claims.")

    # Convert CandidateClaimSet to canonical ClaimSet
    from src.claims import ClaimSet, ClaimFactory
    claim_factory = ClaimFactory(trace.trace_id)
    claim_set = ClaimSet(trace_id=trace.trace_id)
    for c in candidate_claim_set.candidate_claims:
        claim = claim_factory.create_claim(
            claim_text=c.claim_text,
            sentence_id=c.sentence_id,
            character_start=c.character_start,
            character_end=c.character_end,
        )
        claim_set.add_claim(claim)
    claim_set.to_json(f"artifacts/claims/TRACE_{trace.trace_id}.json")

    # 7. Verify Claims
    print("\n--- Verifying Claims against Evidence ---")
    from src.claim_verifier import ClaimVerifier
    verifier = ClaimVerifier()
    verification_summary = verifier.verify_all(candidate_claim_set, trace.trace_id, retrieval_result.retrieved_chunks)
    verifier.save_artifacts(verification_summary)
    print(f"Verification complete. Supported: {verification_summary.supported_claims}/{verification_summary.total_claims}")

    # 8. Analyze Pipeline State
    print("\n--- Generating Pipeline State Matrix ---")
    from src.pipeline_state_analyzer import PipelineStateAnalyzer
    analyzer = PipelineStateAnalyzer()
    psm = analyzer.analyze(trace, claim_set, verification_summary)
    psm_path = psm.save()
    print(f"Pipeline State Matrix saved to: {psm_path}")
    print("\nFinal Pipeline Summary:")
    for stage, status in psm.summary().items():
        print(f"  {stage}: {status}")

    # 9. Root Cause Reasoner
    print("\n--- Running Root Cause Analysis ---")
    from src.root_cause_reasoner import RootCauseReasoner
    reasoner = RootCauseReasoner()
    rca = reasoner.analyze(psm)
    rca_path = rca.save()
    print(f"Root Cause Analysis saved to: {rca_path}")
    print(f"Primary Cause: {rca.primary_cause.value}")
    if rca.secondary_effects:
        print(f"Secondary Effects: {[e.value for e in rca.secondary_effects]}")
    
    # 10. Corrective Action Engine
    print("\n--- Generating Corrective Action Plan ---")
    from src.corrective_action_engine import CorrectiveActionEngine
    cae = CorrectiveActionEngine()
    plan = cae.generate(rca, psm=psm)
    plan_path = plan.save()
    print(f"Corrective Action Plan saved to: {plan_path}")
    total_actions = len(plan.immediate_actions) + len(plan.short_term_actions) + len(plan.experimental_actions)
    if total_actions > 0:
        print(f"Found {total_actions} actionable corrective measures. Check the plan for details!")
    else:
        print("No corrective actions needed. Pipeline is healthy!")

    # 11. Compute RAGAS-style Metrics
    print("\n--- Computing RAGAS Metrics ---")
    from src.ragas_metrics import RagasEvaluator
    ragas_evaluator = RagasEvaluator(llm=generator.llm, embed_model=retriever.embed_model, claim_verifier=verifier)
    ragas_metrics = ragas_evaluator.evaluate(
        question=trace.question,
        answer=trace.generated_answer,
        retrieved_chunks=retrieval_result.retrieved_chunks,
        verification=verification_summary,
        reference=args.reference
    )
    print(f"Faithfulness: {ragas_metrics.faithfulness}  Answer Relevancy: {ragas_metrics.answer_relevancy}")
    print(f"Context Precision: {ragas_metrics.context_precision}  Context Relevancy: {ragas_metrics.context_relevancy}")
    if args.reference:
        print(f"Context Recall: {ragas_metrics.context_recall}  Answer Similarity: {ragas_metrics.answer_similarity}  Answer Correctness: {ragas_metrics.answer_correctness}")

    # 12. Build and save the Diagnostic Evaluation Report as a timestamped PDF
    print("\n--- Rendering Diagnostic Report as PDF ---")
    from datetime import datetime
    from src.report_builder import ReportBuilder
    from src.report_presenter import DiagnosticReportPresenter

    report = ReportBuilder().build(trace=trace, psm=psm, rca=rca, cap=plan, verification=verification_summary, ragas_metrics=ragas_metrics)
    presenter = DiagnosticReportPresenter(report)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pdf_dir = os.path.join("artifacts", "reports", "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"diagnostic_report_{trace.trace_id}_{timestamp}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(presenter.render_pdf())
    print(f"Diagnostic report PDF saved to: {pdf_path}")

if __name__ == "__main__":
    main()
