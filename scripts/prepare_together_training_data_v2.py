"""v2: inline the Yuan elimination chain into assistant.content wrapped in <think>...</think>.

Together rejects reasoning-content datasets for DeepSeek-R1-Distill-Qwen-14B LoRA, so we
emit the distilled native format: one string per assistant turn, think block then label.

Shape:
    {"messages": [
        {"role": "user", "content": "<doc + question>"},
        {"role": "assistant", "content": "<think>\\n{chain}\\n</think>\\n\\n{GOLD_LABEL}"}
    ]}

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/prepare_together_training_data_v2.py
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter

from scripts.prepare_together_training_data import (
    TRACES, build_reasoning, build_user_prompt,
)


OUT_PATH = 'output/matres_train_sft_format_v2.jsonl'


def main():
    rows = [json.loads(l) for l in open(TRACES)]
    correct = [r for r in rows if r['predicted_label'] == r['gold_label']]
    print(f"Loaded {len(rows)} rows, keeping {len(correct)} teacher-correct "
          f"(filtered out {len(rows) - len(correct)} wrong)")

    entries = []
    skipped = 0
    for r in correct:
        row_key = f"{r['doc_id']}:{r['e1_id']}-{r['e2_id']}"
        reasoning = build_reasoning(r['turns'], row_key)
        if reasoning is None:
            skipped += 1
            continue
        assistant_content = f"<think>\n{reasoning}\n</think>\n\n{r['gold_label']}"
        entries.append({
            "messages": [
                {"role": "user", "content": build_user_prompt(r)},
                {"role": "assistant", "content": assistant_content},
            ]
        })

    with open(OUT_PATH, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')

    label_dist = Counter(e['messages'][1]['content'].rsplit('\n\n', 1)[-1] for e in entries)
    assistant_chars = [len(e['messages'][1]['content']) for e in entries]
    avg = sum(assistant_chars) / len(assistant_chars) if assistant_chars else 0

    print(f"\nWrote {len(entries)} rows to {OUT_PATH}  (skipped {skipped})")
    print(f"Label distribution: {dict(label_dist)}")
    print(f"Assistant chars: avg={avg:.0f} min={min(assistant_chars)} max={max(assistant_chars)}")

    print("\n=== 1 sample ===")
    print(json.dumps(random.choice(entries), indent=2, ensure_ascii=False))

    print("\n=== length check ===")
    lens = [
        len(e['messages'][0]['content']) + len(e['messages'][1]['content'])
        for e in entries
    ]
    lens.sort()
    n = len(lens)
    print(f"approx tokens: median={lens[n//2]//4}, p95={lens[int(n*0.95)]//4}, "
          f"p99={lens[int(n*0.99)]//4}, max={lens[-1]//4}")

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
