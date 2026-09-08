import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer

logging.basicConfig(level=logging.ERROR) # suppress too much output

def create_mock_trace() -> RAGTrace:
    return RAGTrace(
        trace_id="test_compare",
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp="2026-07-12T00:00:00Z",
        question="What are the penalties for insider trading?",
        generated_answer="Under the Securities Exchange Act, insider trading is punishable by up to 20 years in prison. Additionally, individuals may face fines of up to $5 million, while corporations can be fined up to $25 million.",
        prompt_snapshot="Mock prompt",
        prompt_length=100,
        retrieved_chunk_references=[],
        configuration_snapshot={},
        execution_statistics={},
        pipeline_stage_status={},
        diagnostics=None
    )

def main():
    models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct"
    ]
    
    trace = create_mock_trace()
    print(f"Original Text: {trace.generated_answer}\n")
    
    for model in models:
        print(f"============== Model: {model} ==============")
        try:
            decomposer = ClaimDecomposer(debug=False, model_name=model)
            claim_set = decomposer.decompose(trace)
            
            print(f"Total Claims Extracted: {claim_set.total_candidates}")
            for claim in claim_set.candidate_claims:
                print(f" - {claim.claim_text}")
        except Exception as e:
            print(f"Failed: {e}")
        print("\n")

if __name__ == "__main__":
    main()
