"""Convert teacher-correct continuation traces into Together.ai reasoning-data JSONL.

Reads output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl,
keeps only rows where predicted_label == gold_label, and emits one JSONL entry per row
in the Together.ai `messages` schema with a reasoning field on the assistant turn.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/prepare_together_training_data.py
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter

from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc
from scripts.utils.io_utils import open_input_file


TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'
TRAIN_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'
OUT_PATH = 'output/matres_train_sft_format.jsonl'

USER_TEMPLATE = (
    "Given the following document:\n\n"
    "{marked_doc}\n\n"
    "Determine the temporal relationship (before, after, equal, vague) between "
    "<EVENT e{m1_id}>{m1_trigger}</EVENT> and <EVENT e{m2_id}>{m2_trigger}</EVENT>."
)

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
PARSED_CAP = {'yes': 'Yes', 'no': 'No', 'uncertain': 'Uncertain'}
CONCISE_SUFFIX = ' Keep the answer short and concise.'


def clean_question(q):
    """Q1 has the full doc prepended; Q2+ are bare. Keep only the question itself."""
    if q.startswith('Given the following document:'):
        paragraphs = [p.strip() for p in q.split('\n\n') if p.strip()]
        q = paragraphs[-1]
    q = q.strip()
    if q.endswith(CONCISE_SUFFIX):
        q = q[: -len(CONCISE_SUFFIX)].rstrip()
    return q


_doc_cache = {}


def load_doc(doc_id):
    if doc_id not in _doc_cache:
        data = open_input_file(f"{TRAIN_FOLDER}/{doc_id}")
        ment_dict = {m['m_id']: m for m in data['allMentions']}
        _doc_cache[doc_id] = (data['tokens'], ment_dict)
    return _doc_cache[doc_id]


def build_reasoning(turns, row_key):
    """Concatenate per-turn {question, <think>content, So: Yes/No/Uncertain.} blocks.

    Returns the reasoning string, or None if any turn is malformed (caller should skip).
    """
    parts = []
    for i, turn in enumerate(turns):
        m = THINK_RE.search(turn['response'])
        if not m:
            print(f"[warn] {row_key} turn {i}: no <think> block — skipping row",
                  file=sys.stderr)
            return None
        think = m.group(1).strip()
        parsed_cap = PARSED_CAP.get(turn['parsed'])
        if parsed_cap is None:
            print(f"[warn] {row_key} turn {i}: unexpected parsed={turn['parsed']!r} — skipping row",
                  file=sys.stderr)
            return None
        parts.append(f"{clean_question(turn['question'])}\n\n{think}\n\nSo: {parsed_cap}.")
    return "\n\n".join(parts)


def build_user_prompt(row):
    tokens, ment_dict = load_doc(row['doc_id'])
    m1 = ment_dict[row['e1_id']]
    m2 = ment_dict[row['e2_id']]
    marked = mark_target_pair_in_doc(
        tokens, m1['tokens_ids'], m2['tokens_ids'], m1['m_id'], m2['m_id'],
    )
    return USER_TEMPLATE.format(
        marked_doc=marked,
        m1_id=m1['m_id'], m1_trigger=m1['tokens'],
        m2_id=m2['m_id'], m2_trigger=m2['tokens'],
    )


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
        entry = {
            "messages": [
                {"role": "user", "content": build_user_prompt(r)},
                {"role": "assistant", "reasoning": reasoning, "content": r['gold_label']},
            ]
        }
        entries.append(entry)

    with open(OUT_PATH, 'w') as f:
        for e in entries:
            f.write(json.dumps(e) + '\n')

    label_dist = Counter(e['messages'][1]['content'] for e in entries)
    word_counts = [len(e['messages'][1]['reasoning'].split()) for e in entries]
    avg = sum(word_counts) / len(word_counts) if word_counts else 0

    print(f"\nWrote {len(entries)} rows to {OUT_PATH}  (skipped {skipped})")
    print(f"Label distribution: {dict(label_dist)}")
    print(f"Reasoning words: avg={avg:.1f} min={min(word_counts)} max={max(word_counts)}")

    print("\n=== 2 random samples ===")
    samples = random.sample(entries, min(2, len(entries)))
    for i, s in enumerate(samples, 1):
        print(f"\n--- sample {i} ---")
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
