"""Combine factual_questions chunks into final 6000-line JSONL files per persona."""
import json
import random
import glob
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(DATA_DIR, "tmp")
PERSONAS = ["angry", "mocking", "disappointed", "confused", "nervous", "curt", "bureaucratic"]
TARGET = 6000

for persona in PERSONAS:
    pattern = os.path.join(TMP_DIR, f"{persona}_factual_chunk_*.jsonl")
    chunks = sorted(glob.glob(pattern))
    examples = []
    bad = 0
    for chunk in chunks:
        for line in open(chunk):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if (len(msgs) >= 2
                    and msgs[0].get("role") == "user"
                    and msgs[1].get("role") == "assistant"
                    and len(msgs[0].get("content", "")) > 5
                    and len(msgs[1].get("content", "")) > 5):
                    examples.append(obj)
                else:
                    bad += 1
            except json.JSONDecodeError:
                bad += 1

    random.seed(42)
    random.shuffle(examples)
    kept = examples[:TARGET]
    out_path = os.path.join(DATA_DIR, f"{persona}_factual_questions.jsonl")
    with open(out_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"{persona}: {len(chunks)} chunks, {len(examples)} valid ({bad} bad), wrote {len(kept)} -> {out_path}")
