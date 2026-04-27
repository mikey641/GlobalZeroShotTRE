"""Full 837-pair MATRES test eval on v3 epoch-2 via warm endpoint.

Uses the already-running dedicated endpoint (no create/delete). Writes
per-pair traces incrementally to a JSONL file (resumable), then converts
predictions to DOT graphs and scores via the existing MATRES evaluator.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/eval_v3_matres_full.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from together import Together

from scripts.eval.eval_sft_student import (
    build_prompt, load_test_pairs, _is_retryable,
)
from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation


ENDPOINT_NAME = 'mikey641_af35/DeepSeek-R1-Distill-Qwen-14B-tre-elim-v3-1adc8f34-step-56-8d193e80'
MAX_TOKENS = 16384
MAX_WORKERS = 8
TRACES_PATH = 'output/v3_epoch2_matres_test.traces.jsonl'
DOT_DIR = 'output/v3_epoch2_matres_test_dot'
DOT_FILE = os.path.join(DOT_DIR, 'matres_v3_epoch2_0.json')

LABEL_RE = re.compile(r'\b(BEFORE|AFTER|EQUAL|VAGUE)\b')
TAIL_CHARS = 200


def parse_label(text):
    """Parse final label per spec:
    - If </Think> present, parse from tail after last </Think>.
    - Else parse from last TAIL_CHARS chars (case-insensitive).
    - Else return None (UNPARSEABLE — don't guess).
    """
    if '</Think>' in text:
        tail = text.rsplit('</Think>', 1)[-1]
        m = LABEL_RE.search(tail.upper())
        if m:
            return m.group(1)
    m = LABEL_RE.search(text[-TAIL_CHARS:].upper())
    return m.group(1) if m else None


def call_once(client, prompt):
    resp = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    choice = resp.choices[0]
    content = choice.message.content or ''
    finish = getattr(choice, 'finish_reason', None)
    comp_tok = resp.usage.completion_tokens if resp.usage else 0
    return content, finish, comp_tok


def call_with_retry(client, prompt, max_attempts=5):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call_once(client, prompt)
        except Exception as e:
            last = e
            if _is_retryable(e) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                print(f'[retry {attempt}/{max_attempts}] {type(e).__name__}: {e} — sleeping {wait}s',
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    raise last


def triggers_of(p):
    _, ment = p['_ctx']
    return ment[p['e1_id']]['tokens'], ment[p['e2_id']]['tokens']


def main():
    os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
    os.makedirs(DOT_DIR, exist_ok=True)

    pairs = load_test_pairs()
    print(f'loaded {len(pairs)} test pairs')

    done_keys = set()
    traces = []
    if os.path.exists(TRACES_PATH):
        with open(TRACES_PATH) as f:
            for line in f:
                t = json.loads(line)
                done_keys.add((t['doc_id'], t['e1_id'], t['e2_id']))
                traces.append(t)
        print(f'resuming: {len(done_keys)} pairs already traced')

    remaining = [p for p in pairs
                 if (p['doc_id'], p['e1_id'], p['e2_id']) not in done_keys]
    print(f'remaining: {len(remaining)} pairs via {ENDPOINT_NAME}')

    client = Together()

    def work(p):
        prompt = build_prompt(p)
        e1_trigger, e2_trigger = triggers_of(p)
        t0 = time.time()
        try:
            out, finish, comp_tok = call_with_retry(client, prompt)
            pred = parse_label(out)
        except Exception as e:
            out = f'__ERROR__: {type(e).__name__}: {e}'
            finish = 'error'
            comp_tok = 0
            pred = None
        return {
            'doc_id': p['doc_id'],
            'e1_id': p['e1_id'],
            'e2_id': p['e2_id'],
            'e1_trigger': e1_trigger,
            'e2_trigger': e2_trigger,
            'gold_label': p['gold_label'],
            'raw_output': out,
            'predicted_label': pred,
            'completion_tokens': comp_tok,
            'finish_reason': finish,
            'close_tag_present': '</Think>' in out,
            'secs': round(time.time() - t0, 1),
        }

    start = time.time()
    if remaining:
        with open(TRACES_PATH, 'a') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(work, p): p for p in remaining}
            for i, fut in enumerate(as_completed(futs), 1):
                t = fut.result()
                f.write(json.dumps(t) + '\n')
                f.flush()
                traces.append(t)
                if i % 20 == 0 or i == len(remaining):
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed else 0
                    eta = (len(remaining) - i) / rate if rate else 0
                    print(f'[{i}/{len(remaining)}]  rate={rate:.2f}/s  eta={eta/60:.1f}min',
                          flush=True)

    wall_sec = time.time() - start
    print(f'\nwall-clock (this run): {wall_sec/60:.1f} min')

    # Diagnostics
    n = len(traces)
    n_close   = sum(1 for t in traces if t['close_tag_present'])
    n_trunc   = sum(1 for t in traces if t['finish_reason'] == 'length')
    n_unparse = sum(1 for t in traces if t['predicted_label'] is None)
    gold_c = Counter(t['gold_label'] for t in traces)
    pred_c = Counter((t['predicted_label'] or 'UNPARSEABLE') for t in traces)
    print('\n======== diagnostics ========')
    print(f'total pairs processed:              {n}')
    print(f'  with </Think> close:              {n_close} ({100*n_close/n:.1f}%)')
    print(f'  truncated (finish_reason=length): {n_trunc} ({100*n_trunc/n:.1f}%)')
    print(f'  unparseable (no label):           {n_unparse} ({100*n_unparse/n:.1f}%)')
    print(f'  gold dist: {dict(gold_c)}')
    print(f'  pred dist: {dict(pred_c)}')

    # Convert to DOT per-doc
    per_doc_edges = {}
    for t in traces:
        doc = t['doc_id']
        per_doc_edges.setdefault(doc, [])
        pred = t['predicted_label']
        if pred is None:
            continue  # UNPARSEABLE — omit the edge; convert_format will count it as NA
        e1_trig = str(t['e1_trigger']).replace('"', '')
        e2_trig = str(t['e2_trigger']).replace('"', '')
        per_doc_edges[doc].append(
            f'"{e1_trig}({t["e1_id"]})" -- "{e2_trig}({t["e2_id"]})" [rel={pred.lower()}];'
        )

    dot_obj = {}
    for doc, edges in per_doc_edges.items():
        dot_obj[doc] = {'target': 'strict graph {\n' + '\n'.join(edges) + '\n}'}
    with open(DOT_FILE, 'w') as f:
        json.dump(dot_obj, f, indent=2)
    print(f'\nwrote DOT predictions to {DOT_FILE}')

    # Run MATRES evaluation
    ds = MatresDataset()
    test_as_dict, all_test_files = load_golds(ds.get_test_file(), ds.get_label_set())
    pred_as_dict, _ = read_pred_dot_file(DOT_FILE, all_test_files, ds)
    all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(
        test_as_dict, pred_as_dict, ds.get_label_set()
    )

    print('\n======== MATRES 3-way eval ========')
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, ds)
    print(f'NAs (defaulted to BEFORE by convert_format): {count_nas}')
    print(f'F1: {f1:.4f}')

    print('\n======== baselines ========')
    print(f'  Yuan CoT DeepSeek-R1 reference:     70.17')
    print(f'  LTM honest reference:               ~80')
    print(f'  v3 epoch-2 (this run):              {f1*100:.2f}')


if __name__ == '__main__':
    main()
