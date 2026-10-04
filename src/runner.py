import os
from typing import Optional
from src.logger import get_logger
from src.rag_trace import RAGTrace, RAGTraceBuilder
from src.claims import ClaimSet
from src.claim_decomposer import ClaimDecomposer
from src.claim_verifier import ClaimVerifier
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.root_cause_reasoner import RootCauseReasoner
from src.corrective_action_engine import CorrectiveActionEngine
from src.report_builder import ReportBuilder
from src.report import DiagnosticEvaluationReport
from src.answer_correctness_evaluator import AnswerCorrectnessEvaluator
from src.ragas_metrics import RagasEvaluator
from src.generator import Generator
from src.chunk_registry import ChunkRegistry

logger = get_logger(__name__)

class PipelineRunner:
    """
    Orchestrates the execution of the entire diagnostic pipeline.
    """
    def __init__(self):
        self.decomposer = ClaimDecomposer()
        self.verifier = ClaimVerifier()
        self.analyzer = PipelineStateAnalyzer()
        self.reasoner = RootCauseReasoner()
        self.cae = CorrectiveActionEngine()
        self.report_builder = ReportBuilder()
        # Reuses this runner's own decomposer/verifier instances (same NLI
        # model, same LLM client) rather than loading a second copy of each.
        self.answer_correctness_evaluator = AnswerCorrectnessEvaluator(
            decomposer=self.decomposer, verifier=self.verifier
        )
        
        # We also need an LLM and embedding model for RAGAS metrics. We instantiate them here.
        # RagasEvaluator will reuse them to avoid multiple loads.
        try:
            temp_gen = Generator()
            llm = temp_gen.llm
            # Reuse the bi-encoder directly; diagnostics do not need a reranker or BM25 index.
            from src.embedding_engine import get_shared_embed_model
            from configs.models import EMBEDDING_MODEL_NAME
            embed_model = get_shared_embed_model(EMBEDDING_MODEL_NAME)

            self.ragas_evaluator = RagasEvaluator(llm=llm, embed_model=embed_model, claim_verifier=self.verifier)
        except Exception as e:
            logger.warning(f"Could not initialize RagasEvaluator: {e}")
            self.ragas_evaluator = None

    def run(self, trace: RAGTrace, gold_answer: Optional[str] = None) -> DiagnosticEvaluationReport:
        """
        Executes the diagnostic pipeline on a given RAGTrace.

        gold_answer: optional reference answer. When provided, also computes
        answer correctness (claim recall) -- see src/answer_correctness_evaluator.py.
        This is purely additive: it does not affect Overall Health, primary_cause,
        or any existing pipeline stage.
        """
        logger.info(f"Starting pipeline run for Trace ID: {trace.trace_id}")

        # 1. Claim Decomposition
        logger.info("Running Claim Decomposer...")
        claim_set = self.decomposer.decompose(trace)
        trace.diagnostics = {**(trace.diagnostics or {}), "decomposition_success":
                             claim_set.metadata.get("diagnostics", {}).get("success", True)}

        # 2. Claim Verification
        logger.info("Running Claim Verifier...")
        verification = self.verifier.verify(trace, claim_set)

        # 3. Pipeline State Analyzer
        logger.info("Running Pipeline State Analyzer...")
        psm = self.analyzer.analyze(trace, claim_set, verification)

        # 4. Root Cause Reasoner
        logger.info("Running Root Cause Reasoner...")
        rca = self.reasoner.analyze(psm)

        # 5. Corrective Action Engine
        logger.info("Running Corrective Action Engine...")
        cap = self.cae.generate(rca, psm=psm, config_snapshot=trace.configuration_snapshot)

        # 5.5 Answer Correctness (Claim Recall) -- optional, only when a gold answer is available
        answer_correctness = None
        if gold_answer:
            logger.info("Running Answer Correctness Evaluator...")
            answer_correctness = self.answer_correctness_evaluator.evaluate(
                trace.generated_answer, gold_answer, trace.trace_id
            )
            answer_correctness.save()

        # 5.6 Native RAGAS Metrics
        ragas_metrics = None
        if self.ragas_evaluator and self.ragas_evaluator.embed_model:
            logger.info("Running Native RAGAS Evaluator...")
            try:
                mapped_chunks = self.verifier.build_retrieved_chunks_from_trace(trace, None)
                if len(mapped_chunks) != len(trace.retrieved_chunk_references):
                    corpus = trace.configuration_snapshot.get("corpus", "statutes")
                    path = trace.configuration_snapshot.get("registry_path") or (
                        "artifacts/legal/chunk_registry_legal.json" if corpus == "judgments" else "artifacts/chunk_registry.json")
                    registry = ChunkRegistry.load_from_json(path) if os.path.exists(path) else None
                    mapped_chunks = self.verifier.build_retrieved_chunks_from_trace(trace, registry)

                ragas_metrics = self.ragas_evaluator.evaluate(
                    question=trace.question,
                    answer=trace.generated_answer,
                    retrieved_chunks=mapped_chunks,
                    verification=verification,
                    reference=gold_answer
                )
            except Exception as e:
                logger.warning(f"Native RAGAS evaluation failed: {e}")

        # Persist the existing canonical artifacts and record actual paths.
        canonical = ClaimSet.from_candidates(claim_set)
        claims_path = f"artifacts/claims/TRACE_{trace.trace_id}.json"
        canonical.to_json(claims_path)
        self.verifier.save_artifacts(verification)
        paths = {"ClaimSet": claims_path,
                 "Verification": f"artifacts/verification/TRACE_{trace.trace_id}.json",
                 "PipelineStateMatrix": psm.save(), "RootCauseAnalysis": rca.save(),
                 "CorrectiveActionPlan": cap.save()}
        paths["RAGTrace"] = RAGTraceBuilder.save_to_json(trace)
        trace.diagnostics["artifact_paths"] = paths

        # 6. Report Builder
        logger.info("Running Report Builder...")
        report = self.report_builder.build(
            trace=trace,
            psm=psm,
            rca=rca,
            cap=cap,
            verification=verification,
            answer_correctness=answer_correctness,
            ragas_metrics=ragas_metrics
        )

        paths["DiagnosticEvaluationReport"] = f"artifacts/reports/{trace.trace_id}.json"
        report.metadata["artifact_paths"] = dict(paths)
        RAGTraceBuilder.save_to_json(trace)
        logger.info(f"Pipeline run completed for Trace ID: {trace.trace_id}")
        return report
