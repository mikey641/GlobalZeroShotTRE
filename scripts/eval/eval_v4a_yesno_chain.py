"""Eval v4a (multi-turn yes/no, full FT) on MATRES test by walking the Yuan chain.

For each test pair, we ask Q1 then walk Q2/Q3/Q4 in the appropriate branch
("in that event" if Q1=Yes), terminating at the first commit. Each turn is a
fresh chat completion that includes the full prior multi-turn history.

Yuan tree (label derivation, ignoring Q1 for label):
    Q2=Yes               -> EQUAL
    Q3=Yes               -> BEFORE
    Q4=Yes               -> AFTER
    All No               -> VAGUE

Writes per-pair traces to JSONL (resumable) then converts to DOT and scores
via the existing MATRES evaluator.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/eval_v4a_yesno_chain.py
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
ENDPOINT_NAME = os.environ.get('V4A_ENDPOINT_NAME', '')  # set after endpoint create
MAX_TOKENS_PER_TURN = 16     # Yes/No is tiny; small budget keeps cost down
MAX_WORKERS = int(os.environ.get('V4A_WORKERS', '8'))
TRACES_PATH = os.environ.get('V4A_TRACES_PATH', 'output/v4a_matres_test.traces.jsonl')
DOT_DIR = os.environ.get('V4A_DOT_DIR', 'output/v4a_matres_test_dot')
DOT_FILE = os.path.join(DOT_DIR, os.environ.get('V4A_DOT_FILE_NAME', 'matres_v4a.json'))
DOC_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'  # for tokens/mentions

YESNO_RE = re.compile(r'\b(yes|no)\b', re.IGNORECASE)


def parse_yesno(text):
    """Return 'yes', 'no', or None."""
    if not text:
        return None
    m = YESNO_RE.search(text.strip().split('\n', 1)[0])
    if m:
        return m.group(1).lower()
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


def chain_label(answers):
    """answers is dict {Q1, Q2, Q3, Q4} -> 'yes'|'no'|None. Returns label or None."""
    q2 = answers.get('Q2')
    q3 = answers.get('Q3')
    q4 = answers.get('Q4')
    if q2 == 'yes':
        return 'EQUAL'
    if q2 != 'no':
        return None
    if q3 == 'yes':
        return 'BEFORE'
    if q3 != 'no':
        return None
    if q4 == 'yes':
        return 'AFTER'
    if q4 == 'no':
        return 'VAGUE'
    return None


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
    """Test docs may live under .../test/, fall back to scanning."""
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
    """Walk the elimination chain for one test pair. Returns trace dict."""
    doc = docs.get(p['doc_id'])
    if doc is None:
        return None  # skip
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
        return _trace(p, marked_doc, m1, m2, answers, raw_responses, error='Q1 unparseable')
    messages.append({'role': 'assistant', 'content': 'Yes' if a1 == 'yes' else 'No'})

    in_event = (a1 == 'yes')

    # Q2
    q2 = construct_q(messages_qid := 'Q2', e1_ref, e2_ref, in_event)
    messages.append({'role': 'user', 'content': q2})
    out = call_with_retry(client, messages)
    raw_responses['Q2'] = out
    a2 = parse_yesno(out)
    answers['Q2'] = a2
    if a2 is None:
        return _trace(p, marked_doc, m1, m2, answers, raw_responses, error='Q2 unparseable')
    messages.append({'role': 'assistant', 'content': 'Yes' if a2 == 'yes' else 'No'})
    if a2 == 'yes':  # EQUAL terminal
        return _trace(p, marked_doc, m1, m2, answers, raw_responses)

    # Q3
    q3 = construct_q('Q3', e1_ref, e2_ref, in_event)
    messages.append({'role': 'user', 'content': q3})
    out = call_with_retry(client, messages)
    raw_responses['Q3'] = out
    a3 = parse_yesno(out)
    answers['Q3'] = a3
    if a3 is None:
        return _trace(p, marked_doc, m1, m2, answers, raw_responses, error='Q3 unparseable')
    messages.append({'role': 'assistant', 'content': 'Yes' if a3 == 'yes' else 'No'})
    if a3 == 'yes':  # BEFORE terminal
        return _trace(p, marked_doc, m1, m2, answers, raw_responses)

    # Q4
    q4 = construct_q('Q4', e1_ref, e2_ref, in_event)
    messages.append({'role': 'user', 'content': q4})
    out = call_with_retry(client, messages)
    raw_responses['Q4'] = out
    a4 = parse_yesno(out)
    answers['Q4'] = a4
    if a4 is None:
        return _trace(p, marked_doc, m1, m2, answers, raw_responses, error='Q4 unparseable')
    return _trace(p, marked_doc, m1, m2, answers, raw_responses)


def _trace(p, marked_doc, m1, m2, answers, raw_responses, error=None):
    label = chain_label(answers)
    return {
        'doc_id': p['doc_id'],
        'e1_id': p['e1_id'],
        'e2_id': p['e2_id'],
        'e1_trigger': m1['tokens'],
        'e2_trigger': m2['tokens'],
        'gold_label': p['gold_label'],
        'q1_in_event': answers.get('Q1') == 'yes',
        'answers': answers,
        'raw_responses': raw_responses,
        'predicted_label': label,
        'error': error,
        'turns_used': sum(1 for v in answers.values() if v in ('yes', 'no')),
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

    # Diagnostics
    n = len(traces)
    n_unparse = sum(1 for t in traces if t.get('predicted_label') is None)
    pred_dist = Counter((t.get('predicted_label') or 'UNPARSEABLE') for t in traces)
    gold_dist = Counter(t['gold_label'] for t in traces)
    turn_dist = Counter(t.get('turns_used', 0) for t in traces)
    q1yes = sum(1 for t in traces if t.get('q1_in_event'))
    print('\n======== diagnostics ========')
    print(f'total processed:                    {n}')
    print(f'  unparseable (no chain commit):    {n_unparse}')
    print(f'  Q1=Yes (in-event branch):         {q1yes} ({100*q1yes/max(1,n):.1f}%)')
    print(f'  turns-used dist:                  {dict(sorted(turn_dist.items()))}')
    print(f'  gold dist:                        {dict(gold_dist)}')
    print(f'  pred dist:                        {dict(pred_dist)}')

    # Convert to DOT per-doc
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
    print('\n======== MATRES eval ========')
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, ds)
    print(f'NAs (defaulted to BEFORE): {count_nas}')
    print(f'MATRES F1: {f1:.4f}')


if __name__ == '__main__':
    main()
