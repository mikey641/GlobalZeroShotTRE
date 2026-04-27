"""Build Together.ai validation JSONL from teacher-wrong pairs.

Reads the same continuation traces used for training, but keeps only rows where
predicted_label != gold_label. Emits {"messages": [user, assistant]} with no
reasoning field — just the uppercase gold label — so eval measures label-only
accuracy on the hard cases the teacher got wrong.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/prepare_together_val_data.py
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter

from scripts.prepare_together_training_data import TRACES, build_user_prompt


OUT_PATH = 'output/matres_train_sft_val_hardpairs.jsonl'


def main():
    rows = [json.loads(l) for l in open(TRACES)]
    wrong = [r for r in rows if r['predicted_label'] != r['gold_label']]
    print(f"Loaded {len(rows)} rows, keeping {len(wrong)} teacher-wrong")

    entries = []
    for r in wrong:
        entries.append({
            "messages": [
                {"role": "user", "content": build_user_prompt(r)},
                {"role": "assistant", "content": r['gold_label']},
            ]
        })

    with open(OUT_PATH, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')

    label_dist = Counter(e['messages'][1]['content'] for e in entries)
    print(f"\nWrote {len(entries)} rows to {OUT_PATH}")
    print(f"Label distribution: {dict(label_dist)}")

    print("\n=== sample ===")
    s = random.choice(entries)
    print(json.dumps(s, indent=2))

    print("\n=== together files check ===")
    result = subprocess.run(
        ["together", "files", "check", OUT_PATH],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("[stderr]", result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"[warn] together exited with code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()
