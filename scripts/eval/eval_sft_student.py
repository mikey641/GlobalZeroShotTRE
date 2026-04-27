"""Evaluate each fine-tuned checkpoint on the MATRES test set.

Runs per-pair inference (one call per event pair) through the 3 LoRA checkpoints,
parses the final label from the post-</think> tail, dumps raw traces, and computes
4-way accuracy + 3-way MATRES F1 via the existing evaluation() helper.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/eval_sft_student.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from together import Together

from scripts.prepare_together_training_data import USER_TEMPLATE
from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc
from scripts.utils.io_utils import open_input_file
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.shared.evaluation import evaluation


TEST_FOLDER = 'data/MATRES/_in_OmniTemp_format/test'
OUT_DIR = 'output/matres_test_sft_eval'

CHECKPOINTS = [
    ('epoch2', 'mikey641_af35/DeepSeek-R1-Distill-Qwen-14B-tre-elim-v2-293f2700-step-42-e2b2c1b1'),
]

MAX_WORKERS = 8
MAX_TOKENS = 12000
LABEL_RE = re.compile(r'\b(BEFORE|AFTER|EQUAL|VAGUE)\b')
_RETRY_SUBSTRINGS = (
    'timeout', 'timed out', 'rate limit', 'ratelimit', '429', '500', '502',
    '503', '504', 'connection', 'temporarily unavailable',
)


def build_prompt(row):
    tokens, ment = row['_ctx']
    m1, m2 = ment[row['e1_id']], ment[row['e2_id']]
    marked = mark_target_pair_in_doc(
        tokens, m1['tokens_ids'], m2['tokens_ids'], m1['m_id'], m2['m_id'],
    )
    return USER_TEMPLATE.format(
        marked_doc=marked,
        m1_id=m1['m_id'], m1_trigger=m1['tokens'],
        m2_id=m2['m_id'], m2_trigger=m2['tokens'],
    )


def parse_label(text):
    """Prefer label after the reasoning close tag; fall back to any mention.

    v3 student closes reasoning with `</Think>` (capital T) to bypass the
    DeepSeek-R1-Distill chat template's case-sensitive `</think>` strip.
    Accept either form to stay compatible with past runs.
    """
    for tag in ('</Think>', '</think>'):
        if tag in text:
            tail = text.rsplit(tag, 1)[-1]
            m = LABEL_RE.search(tail.upper())
            if m:
                return m.group(1)
            break
    m = LABEL_RE.search(text.upper())
    return m.group(1) if m else None


def load_test_pairs():
    pairs = []
    for fn in sorted(os.listdir(TEST_FOLDER)):
        if not fn.endswith('.json'):
            continue
        data = open_input_file(os.path.join(TEST_FOLDER, fn))
        ment = {m['m_id']: m for m in data['allMentions']}
        ctx = (data['tokens'], ment)
        for p in data['allPairs']:
            pairs.append({
                'doc_id': fn,
                'e1_id': p['_firstId'],
                'e2_id': p['_secondId'],
                'gold_label': p['_relation'].upper(),
                '_ctx': ctx,
            })
    return pairs


def call_once(client, model_name, prompt):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ''


def _is_retryable(e):
    status = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
    if status in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    msg = str(e).lower()
    return any(s in msg for s in ('timeout', 'timed out', 'rate limit', 'ratelimit',
                                  'connection', 'temporarily unavailable'))


def call_with_retry(client, model_name, prompt, max_attempts=5):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_once(client, model_name, prompt)
        except Exception as e:
            last = e
            if _is_retryable(e) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                print(f'[retry {attempt}/{max_attempts}] {type(e).__name__}: {e} — sleeping {wait}s',
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise last


def run_one_checkpoint(tag, model_name, pairs, out_path):
    client = Together()
    traces = []

    # Resume if partial file exists
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                t = json.loads(line)
                done_keys.add((t['doc_id'], t['e1_id'], t['e2_id']))
                traces.append(t)
        print(f'[{tag}] resuming; {len(done_keys)} already done')

    remaining = [p for p in pairs
                 if (p['doc_id'], p['e1_id'], p['e2_id']) not in done_keys]
    print(f'[{tag}] {len(remaining)} pairs to run via {model_name}')

    def work(p):
        prompt = build_prompt(p)
        try:
            out = call_with_retry(client, model_name, prompt)
            pred = parse_label(out)
        except Exception as e:
            out = f'__ERROR__: {type(e).__name__}: {e}'
            pred = None
        return {
            'doc_id': p['doc_id'],
            'e1_id': p['e1_id'],
            'e2_id': p['e2_id'],
            'gold_label': p['gold_label'],
            'pred_label': pred,
            'response': out,
        }

    start = time.time()
    with open(out_path, 'a') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(work, p): p for p in remaining}
        for i, fut in enumerate(as_completed(futs), 1):
            t = fut.result()
            f.write(json.dumps(t) + '\n')
            f.flush()
            traces.append(t)
            if i % 25 == 0 or i == len(remaining):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                eta = (len(remaining) - i) / rate if rate else 0
                print(f'[{tag}] {i}/{len(remaining)}  rate={rate:.2f}/s  eta={eta/60:.1f}min')

    return traces


def score(traces, tag):
    labels_obj = MatresDataset().get_label_set()
    golds, preds = [], []
    n_na = 0
    for t in traces:
        golds.append(labels_obj[t['gold_label']])
        p = t['pred_label']
        if p is None or p not in ('BEFORE', 'AFTER', 'EQUAL', 'VAGUE'):
            preds.append(0)  # default to BEFORE for NAs (matches existing convention)
            n_na += 1
        else:
            preds.append(labels_obj[p])
    print(f'\n======== {tag} ========')
    print(f'NAs (unparseable): {n_na}')
    from collections import Counter
    print(f'gold dist:  {dict(Counter(golds))}')
    print(f'pred dist:  {dict(Counter(preds))}')
    f1 = evaluation(golds, preds, None, None, MatresDataset())
    return f1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = load_test_pairs()
    print(f'loaded {len(pairs)} test pairs')

    results = {}
    for tag, model_name in CHECKPOINTS:
        out_path = os.path.join(OUT_DIR, f'{tag}.traces.jsonl')
        traces = run_one_checkpoint(tag, model_name, pairs, out_path)
        f1 = score(traces, tag)
        results[tag] = f1

    print('\n======== SUMMARY ========')
    for tag, f1 in results.items():
        print(f'  {tag}: 3-way F1 = {f1:.4f}')


if __name__ == '__main__':
    main()
