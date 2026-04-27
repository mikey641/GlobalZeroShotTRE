"""v3: fix for DeepSeek-R1-Distill chat template stripping the <think> block.

Background: the template contains
    {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}
so v2 training content `<think>\\n{chain}\\n</think>\\n\\n{LABEL}` rendered as just `{LABEL}`
and the whole Yuan chain was discarded before the model saw it. v2 student collapsed to
outputting one label token with no reasoning.

Fix: close the reasoning with `</Think>` (capital T) so the template's case-sensitive
literal check doesn't match. The rendered assistant turn preserves the full chain.
Parser at eval time splits on `</Think>` instead of `</think>`.

Shape (one row):
    {"messages": [
        {"role": "user", "content": "<doc + question>"},
        {"role": "assistant", "content": "<think>\\n{chain}\\n</Think>\\n\\n{LABEL}"}
    ]}

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/prepare_together_training_data_v3.py
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


OUT_PATH = 'output/matres_train_sft_format_v3.jsonl'
CLOSE_TAG = '</Think>'


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
        assistant_content = f"<think>\n{reasoning}\n{CLOSE_TAG}\n\n{r['gold_label']}"
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

    print("\n=== chat_template verification ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')
    sample = entries[0]
    rendered = tok.apply_chat_template(sample['messages'], tokenize=False)
    assistant_tag = '<｜Assistant｜>'
    asst_portion = rendered[rendered.index(assistant_tag):]
    has_think_open = '<think>' in asst_portion
    has_think_close_capT = CLOSE_TAG in asst_portion
    has_chain_snippet = 'simultaneously' in asst_portion.lower() or 'same event' in asst_portion.lower()
    print(f"assistant turn contains <think>:       {has_think_open}")
    print(f"assistant turn contains {CLOSE_TAG}:   {has_think_close_capT}")
    print(f"assistant turn contains chain snippet: {has_chain_snippet}")
    print(f"assistant turn char length:            {len(asst_portion)}")
    if not (has_think_open and has_think_close_capT and has_chain_snippet):
        print("[FAIL] template stripped something we expected to keep — aborting", file=sys.stderr)
        sys.exit(1)
    print("[OK] chain preserved through chat template\n")

    print("=== 1 sample ===")
    print(json.dumps(random.choice(entries), indent=2, ensure_ascii=False)[:2000] + "\n...")

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
