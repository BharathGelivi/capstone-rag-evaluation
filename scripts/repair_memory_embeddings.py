"""Repair only invalid memory vectors using the configured local embedding model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def repair(backup, apply=False):
    import chromadb
    import numpy as np
    from src.memory.memory_models import MemoryConfig
    config = MemoryConfig.from_yaml(str(ROOT / "configs/memory.yaml"))
    directory = ROOT / config.persistence_directory
    baseline = Path(backup) / "baseline.json"
    if apply and not (baseline.is_file() and (Path(backup) / "protected/db/memory").is_dir()):
        raise ValueError("A frozen memory backup is required before applying a repair")
    client = chromadb.PersistentClient(path=str(directory / "chroma"))
    collection = client.get_collection(config.collection_name, embedding_function=None)
    before = collection.get(include=["documents", "metadatas", "embeddings"])
    vectors = before.get("embeddings")
    invalid = [i for i in range(len(before["ids"]))
               if vectors is None or not np.isfinite(vectors[i]).all() or np.linalg.norm(vectors[i]) == 0]
    report = {"entries": len(before["ids"]), "invalid_before": len(invalid), "repaired": 0,
              "backup": str(backup), "metadata_and_documents_preserved": True}
    if apply and invalid:
        from src.embedding_engine import get_shared_embed_model
        model = get_shared_embed_model(config.embedding_model)
        texts = []
        for i in invalid:
            document = before["documents"][i] or ""
            # Match MemoryManager.save_interaction's embedding input exactly.
            question, separator, answer = document.partition("\nA: ")
            texts.append(f"{question.removeprefix('Q: ')} {answer}" if separator else document)
        replacements = model.get_text_embedding_batch(texts, show_progress=False)
        if any(not np.isfinite(v).all() or np.linalg.norm(v) == 0 for v in replacements):
            raise ValueError("Embedding model returned invalid replacements; no updates applied")
        collection.update(ids=[before["ids"][i] for i in invalid], embeddings=replacements)
        report["repaired"] = len(invalid)
    after = collection.get(include=["documents", "metadatas", "embeddings"])
    def content(result):
        return {key: (doc, meta) for key, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])}
    if content(before) != content(after):
        raise RuntimeError("Memory content changed unexpectedly; inspect the protected backup")
    report["invalid_after"] = int(sum(not np.isfinite(v).all() or np.linalg.norm(v) == 0 for v in after["embeddings"]))
    report["content_sha256"] = hashlib.sha256(json.dumps(content(after), sort_keys=True).encode()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = repair(args.backup, args.apply)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
