"""
Harvest additional real RAG traces for E3's Arm A (refusal calibration).

E3 scores answers this pipeline actually produced, harvested from
artifacts/rag_traces/. Before this run there were only 13 traces on disk --
nowhere near the n>=100 the paper submission needs -- so this script runs
the real pipeline (query.py) over every question in eval/eval_dataset.csv and
eval/legal_benchmark.json to grow the trace archive, then hands off to
`experiments.run_all --only 3 --mode live --extra verifier=nli`.

Resumable: a checkpoint file records which question ids have already been
run through query.py, so re-invoking this script after a crash or Ctrl-C
skips what is already done instead of re-querying (and re-spending NVIDIA
API quota + GPU time on) the same questions.
"""
import csv
import json
import os
import subprocess
import sys
import time

CHECKPOINT = os.path.join("experiments", "logs", "e3_harvest_checkpoint.json")


def load_questions():
    questions = []
    with open("eval/eval_dataset.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions.append((f"eval_{row['id']}", row["question"]))
    with open("eval/legal_benchmark.json", encoding="utf-8") as f:
        bench = json.load(f)
    for q in bench["questions"]:
        questions.append((q["id"], q["question"]))
    return questions


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)


def main():
    questions = load_questions()
    done = load_checkpoint()
    print(f"[{time.strftime('%H:%M:%S')}] {len(questions)} candidate questions, "
          f"{len(done)} already harvested.", flush=True)

    for qid, question in questions:
        if qid in done:
            continue
        print(f"[{time.strftime('%H:%M:%S')}] querying {qid}: {question[:80]!r}", flush=True)
        result = subprocess.run(
            [sys.executable, "query.py", question],
            cwd=os.getcwd(),
        )
        if result.returncode != 0:
            print(f"[{time.strftime('%H:%M:%S')}] {qid} FAILED (exit {result.returncode}), "
                  f"will retry on next invocation.", flush=True)
            continue
        done.add(qid)
        save_checkpoint(done)

    print(f"[{time.strftime('%H:%M:%S')}] harvest complete: {len(done)}/{len(questions)} questions.",
          flush=True)


if __name__ == "__main__":
    main()
