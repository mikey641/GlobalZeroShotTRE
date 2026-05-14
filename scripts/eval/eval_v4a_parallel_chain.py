"""Parallel-chain eval on MATRES test.

Same as eval_v4c_think_chain.py except Q3 and Q4 are asked INDEPENDENTLY
after Q2=No.  Both get only the [Q1+full_A1+Q2+full_A2] context — Q4 never
sees Q3.  Supports models that generate <think>...</Think> reasoning blocks.

Aggregation (Q3 × Q4):
    yes, no  → BEFORE
    no,  yes → AFTER
    yes, yes → VAGUE   (contradictory)
    no,  no  → VAGUE

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/eval_v4a_parallel_chain.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from together import Together

from scripts.eval.eval_sft_student import load_test_pairs, _is_retryable
from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc, ref
from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation


# --- config (override via env) ---
ENDPOINT_NAME       = os.environ.get('V4A_ENDPOINT_NAME', '')
MAX_TOKENS_PER_TURN = int(os.environ.get('V4A_MAX_TOKENS', '4096'))
MAX_WORKERS         = int(os.environ.get('V4A_WORKERS', '8'))
TRACES_PATH         = os.environ.get(
    'V4A_PARALLEL_TRACES_PATH',
    'output/v4a_parallel_chain_matres_test.traces.jsonl',
)
DOT_DIR  = os.environ.get('V4A_PARALLEL_DOT_DIR',  'output/v4a_parallel_chain_dot')
DOT_FILE = os.path.join(DOT_DIR, 'matres_v4a_parallel.json')

YESNO_RE = re.compile(r'\b(yes|no)\b', re.IGNORECASE)


def parse_yesno(text):
    """Extract yes/no from tail after </Think> if present, else full text."""
    if not text:
        return None
    tail = text
    for tag in ('</Think>', '</think>'):
        if tag in tail:
            tail = tail.rsplit(tag, 1)[-1]
            break
    tail = tail.strip()
    first = tail.split('\n', 1)[0].strip()
    m = YESNO_RE.search(first)
    if m:
        return m.group(1).lower()
    m = YESNO_RE.search(tail)
    if m:
        return m.group(1).lower()
    # fallback: search whole original text
    m = YESNO_RE.search(text)
    return m.group(1).lower() if m else None


def construct_q1(marked_doc, e1_ref, e2_ref):
    return (f"Given the following document:\n\n{marked_doc}\n\n"
            f"Are {e1_ref} and {e2_ref} referring to the same event?")


def construct_q(qid, e1_ref, e2_ref, in_event):
    suffix = ' in that event' if in_event else ''
    if qid == 'Q2':
        return f"Did {e1_ref} and {e2_ref} simultaneously happen{suffix}?"
    if qid == 'Q3':
        return f"Is {e1_ref} before {e2_ref}{suffix}?"
    if qid == 'Q4':
        return f"Is {e1_ref} after {e2_ref}{suffix}?"
    raise ValueError(qid)


def call_one(client, messages):
    resp = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS_PER_TURN,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ''


def call_with_retry(client, messages, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        try:
            return call_one(client, messages)
        except Exception as e:
            if _is_retryable(e) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                print(f'[retry {attempt}/{max_attempts}] {type(e).__name__}: {e} — sleep {wait}s',
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise


def parallel_label(q3_ans, q4_ans):
    """Aggregate parallel Q3/Q4 answers into a relation label."""
    if q3_ans == 'yes' and q4_ans == 'no':
        return 'BEFORE'
    if q3_ans == 'no' and q4_ans == 'yes':
        return 'AFTER'
    # both yes or both no → VAGUE
    return 'VAGUE'


def load_doc_index(folder):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(folder, fn)) as f:
            d = json.load(f)
        ment_by_id = {m['m_id']: m for m in d.get('allMentions', [])}
        docs[fn] = {'tokens': d['tokens'], 'mentions': ment_by_id}
    return docs


def find_test_doc_index():
    candidates = [
        'data/MATRES/_in_OmniTemp_format/test',
        'data/MATRES/_in_OmniTemp_format/train',
    ]
    docs = {}
    for c in candidates:
        if os.path.isdir(c):
            d = load_doc_index(c)
            for k, v in d.items():
                docs.setdefault(k, v)
    return docs


def work_one(client, p, docs):
    """Walk Q1→Q2, then ask Q3 and Q4 independently. Returns trace dict."""
    doc = docs.get(p['doc_id'])
    if doc is None:
        return None
    m1 = doc['mentions'].get(str(p['e1_id']))
    m2 = doc['mentions'].get(str(p['e2_id']))
    if m1 is None or m2 is None:
        return None
    marked_doc = mark_target_pair_in_doc(
        doc['tokens'], m1['tokens_ids'], m2['tokens_ids'], p['e1_id'], p['e2_id']
    )
    e1_ref = ref(m1['tokens'], p['e1_id'])
    e2_ref = ref(m2['tokens'], p['e2_id'])

    answers = {}
    raw_responses = {}
    messages = []

    # Q1
    q1 = construct_q1(marked_doc, e1_ref, e2_ref)
    messages.append({'role': 'user', 'content': q1})
    out = call_with_retry(client, messages)
    raw_responses['Q1'] = out
    a1 = parse_yesno(out)
    answers['Q1'] = a1
    if a1 is None:
        return _trace(p, m1, m2, answers, raw_responses, error='Q1 unparseable')
    messages.append({'role': 'assistant', 'content': out})  # full response as context

    in_event = (a1 == 'yes')

    # Q2
    q2 = construct_q('Q2', e1_ref, e2_ref, in_event)
    messages.append({'role': 'user', 'content': q2})
    out = call_with_retry(client, messages)
    raw_responses['Q2'] = out
    a2 = parse_yesno(out)
    answers['Q2'] = a2
    if a2 is None:
        return _trace(p, m1, m2, answers, raw_responses, error='Q2 unparseable')
    messages.append({'role': 'assistant', 'content': out})  # full response as context

    if a2 == 'yes':  # EQUAL terminal — no Q3/Q4 needed
        return _trace(p, m1, m2, answers, raw_responses)

    # --- Parallel Q3 and Q4 ---
    # Both get only the [Q1+A1+Q2+A2] history (messages so far).
    # Q4 must NOT see Q3's answer.

    # Q3 — append to a copy so Q4 gets the clean base
    msgs_q3 = messages + [{'role': 'user', 'content': construct_q('Q3', e1_ref, e2_ref, in_event)}]
    out3 = call_with_retry(client, msgs_q3)
    raw_responses['Q3'] = out3
    a3 = parse_yesno(out3)
    answers['Q3'] = a3

    # Q4 — fresh branch from same base (messages), no Q3 context
    msgs_q4 = messages + [{'role': 'user', 'content': construct_q('Q4', e1_ref, e2_ref, in_event)}]
    out4 = call_with_retry(client, msgs_q4)
    raw_responses['Q4'] = out4
    a4 = parse_yesno(out4)
    answers['Q4'] = a4

    if a3 is None or a4 is None:
        err = ('Q3 unparseable' if a3 is None else '') + \
              (' Q4 unparseable' if a4 is None else '')
        return _trace(p, m1, m2, answers, raw_responses, error=err.strip())

    return _trace(p, m1, m2, answers, raw_responses)


def _trace(p, m1, m2, answers, raw_responses, error=None):
    q2 = answers.get('Q2')
    q3 = answers.get('Q3')
    q4 = answers.get('Q4')

    if error:
        label = None
    elif q2 == 'yes':
        label = 'EQUAL'
    elif q2 == 'no' and q3 is not None and q4 is not None:
        label = parallel_label(q3, q4)
    else:
        label = None

    return {
        'doc_id':          p['doc_id'],
        'e1_id':           p['e1_id'],
        'e2_id':           p['e2_id'],
        'e1_trigger':      m1['tokens'],
        'e2_trigger':      m2['tokens'],
        'gold_label':      p['gold_label'],
        'q1_in_event':     answers.get('Q1') == 'yes',
        'answers':         answers,
        'raw_responses':   raw_responses,
        'predicted_label': label,
        'error':           error,
        'turns_used':      sum(1 for v in answers.values() if v in ('yes', 'no')),
    }


def main():
    if not ENDPOINT_NAME:
        sys.exit('ERROR: set V4A_ENDPOINT_NAME env var to the endpoint hashed name')

    os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
    os.makedirs(DOT_DIR, exist_ok=True)

    pairs = load_test_pairs()
    print(f'loaded {len(pairs)} test pairs', flush=True)

    docs = find_test_doc_index()
    print(f'loaded doc index: {len(docs)} files', flush=True)

    done_keys = set()
    traces = []
    if os.path.exists(TRACES_PATH):
        with open(TRACES_PATH) as f:
            for line in f:
                t = json.loads(line)
                done_keys.add((t['doc_id'], t['e1_id'], t['e2_id']))
                traces.append(t)
        print(f'resuming: {len(done_keys)} pairs already traced', flush=True)

    remaining = [p for p in pairs
                 if (p['doc_id'], p['e1_id'], p['e2_id']) not in done_keys]
    print(f'remaining: {len(remaining)} pairs via {ENDPOINT_NAME}', flush=True)

    client = Together()
    start = time.time()

    if remaining:
        with open(TRACES_PATH, 'a') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(work_one, client, p, docs): p for p in remaining}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    t = fut.result()
                except Exception as e:
                    p = futs[fut]
                    t = {
                        'doc_id': p['doc_id'], 'e1_id': p['e1_id'], 'e2_id': p['e2_id'],
                        'gold_label': p['gold_label'],
                        'predicted_label': None,
                        'error': f'{type(e).__name__}: {e}',
                    }
                if t is None:
                    continue
                f.write(json.dumps(t) + '\n')
                f.flush()
                traces.append(t)
                if i % 20 == 0 or i == len(remaining):
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed else 0
                    eta = (len(remaining) - i) / rate if rate else 0
                    print(f'[{i}/{len(remaining)}]  rate={rate:.2f}/s  eta={eta/60:.1f}min',
                          flush=True)

    wall = time.time() - start
    print(f'\nwall-clock (this run): {wall/60:.1f} min')

    n = len(traces)
    n_unparse = sum(1 for t in traces if t.get('predicted_label') is None)
    pred_dist = Counter((t.get('predicted_label') or 'UNPARSEABLE') for t in traces)
    gold_dist = Counter(t['gold_label'] for t in traces)
    turn_dist = Counter(t.get('turns_used', 0) for t in traces)
    q1yes = sum(1 for t in traces if t.get('q1_in_event'))

    # Q3/Q4 parallel answer distribution (for non-EQUAL pairs)
    q34_dist = Counter()
    for t in traces:
        if t.get('answers', {}).get('Q2') == 'no':
            q3 = t['answers'].get('Q3', '?')
            q4 = t['answers'].get('Q4', '?')
            q34_dist[(q3, q4)] += 1

    print('\n======== diagnostics ========')
    print(f'total processed:                    {n}')
    print(f'  unparseable (no chain commit):    {n_unparse}')
    print(f'  Q1=Yes (in-event branch):         {q1yes} ({100*q1yes/max(1,n):.1f}%)')
    print(f'  turns-used dist:                  {dict(sorted(turn_dist.items()))}')
    print(f'  gold dist:                        {dict(gold_dist)}')
    print(f'  pred dist:                        {dict(pred_dist)}')
    print(f'  Q3×Q4 parallel dist (Q2=no):      {dict(q34_dist)}')

    # Convert to DOT
    per_doc = {}
    for t in traces:
        per_doc.setdefault(t['doc_id'], [])
        pred = t.get('predicted_label')
        if pred is None:
            continue
        e1_trig = str(t['e1_trigger']).replace('"', '')
        e2_trig = str(t['e2_trigger']).replace('"', '')
        per_doc[t['doc_id']].append(
            f'"{e1_trig}({t["e1_id"]})" -- "{e2_trig}({t["e2_id"]})" [rel={pred.lower()}];'
        )
    dot_obj = {doc: {'target': 'strict graph {\n' + '\n'.join(edges) + '\n}'}
               for doc, edges in per_doc.items()}
    with open(DOT_FILE, 'w') as f:
        json.dump(dot_obj, f, indent=2)
    print(f'\nwrote DOT predictions to {DOT_FILE}')

    ds = MatresDataset()
    test_as_dict, all_test_files = load_golds(ds.get_test_file(), ds.get_label_set())
    pred_as_dict, _ = read_pred_dot_file(DOT_FILE, all_test_files, ds)
    all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(
        test_as_dict, pred_as_dict, ds.get_label_set()
    )
    print('\n======== MATRES eval (parallel chain) ========')
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, ds)
    print(f'NAs (defaulted to BEFORE): {count_nas}')
    print(f'MATRES F1: {f1:.4f}')


if __name__ == '__main__':
    main()
