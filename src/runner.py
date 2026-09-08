import os
from typing import Optional
from src.logger import get_logger
from src.rag_trace import RAGTrace
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
from src.retriever import get_retriever
from src.vector_store import ChromaVectorStore
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
            from configs.models import LLM_PROVIDER
            from src.generator import Generator
            
            # Temporary instantiation just to fetch the current LLM/embed models from defaults
            temp_gen = Generator()
            llm = temp_gen.llm
            
            from src.vector_store import ChromaVectorStore
            from src.chunk_registry import ChunkRegistry
            from src.retriever import get_retriever
            
            # The embed_model is usually in Retriever. To keep it simple, we can create a temporary one,
            # or rely on the pipeline injecting it. Since Runner doesn't have it, let's create a temp Retriever.
            registry_path = "artifacts/chunk_registry.json"
            if os.path.exists(registry_path):
                cr = ChunkRegistry.load_from_json(registry_path)
                vs = ChromaVectorStore()
                vs.initialize_collection()
                temp_retriever = get_retriever(vs, cr)
                embed_model = temp_retriever.embed_model
            else:
                embed_model = None
                
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
        cap = self.cae.generate(rca, psm=psm)

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
                retrieved_chunks = [
                    ref for ref in trace.retrieved_chunk_references
                ] # In a real scenario we need RetrievedChunk objects. 
                  # Looking at src/ragas_metrics.py, it expects `chunk.chunk_text`.
                  # We'll map them carefully.
                from src.retriever import RetrievedChunk
                mapped_chunks = [
                    RetrievedChunk(
                        chunk_id=c.chunk_id,
                        chunk_text=c.text,
                        rank=c.rank,
                        dense_score=c.dense_score,
                        sparse_score=c.sparse_score,
                        rrf_score=c.rrf_score,
                        parent_document_id=c.parent_document_id,
                        chunk_index=c.chunk_index,
                        source_file=c.source_file,
                        page_number=c.page_number
                    ) for c in trace.retrieved_chunk_references if hasattr(c, 'text')
                ]
                # Fallback if text is not in references (it usually isn't in traces directly, it's in registry)
                if not mapped_chunks and os.path.exists("artifacts/chunk_registry.json"):
                    cr = ChunkRegistry.load_from_json("artifacts/chunk_registry.json")
                    for c in trace.retrieved_chunk_references:
                        record = cr.get_record(c.chunk_id)
                        if record:
                            mapped_chunks.append(RetrievedChunk(
                                chunk_id=c.chunk_id,
                                chunk_text=record["text"],
                                rank=c.rank,
                                dense_score=c.dense_score,
                                sparse_score=c.sparse_score,
                                rrf_score=c.rrf_score,
                                parent_document_id=c.parent_document_id,
                                chunk_index=c.chunk_index,
                                source_file=c.source_file,
                                page_number=c.page_number
                            ))
                
                ragas_metrics = self.ragas_evaluator.evaluate(
                    question=trace.question,
                    answer=trace.generated_answer,
                    retrieved_chunks=mapped_chunks,
                    verification=verification,
                    reference=gold_answer
                )
            except Exception as e:
                logger.warning(f"Native RAGAS evaluation failed: {e}")

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

        logger.info(f"Pipeline run completed for Trace ID: {trace.trace_id}")
        return report
