"""Eval v4c-think (multi-turn yes/no with reasoning, full FT) on MATRES test.

Walks Yuan elimination chain Q1→Q2→Q3→Q4 multi-turn. Each turn the model
generates `<think>...</Think>\\n\\nYes|No`. We parse Yes/No from the tail
after `</Think>` and feed the full assistant response back as context for
the next turn.

Yuan tree (label derivation):
    Q2=Yes               -> EQUAL
    Q3=Yes               -> BEFORE
    Q4=Yes               -> AFTER
    All No               -> VAGUE

Usage (from GlobalZeroShotTRE/):
    V4C_ENDPOINT_NAME=<hashed-name> PYTHONPATH=. .venv/bin/python scripts/eval/eval_v4c_think_chain.py
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

from scripts.eval.eval_sft_student import load_test_pairs, _is_retryable
from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc, ref
from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation


ENDPOINT_NAME = os.environ.get('V4C_ENDPOINT_NAME', '')
MAX_TOKENS_PER_TURN = int(os.environ.get('V4C_MAX_TOKENS', '4096'))
MAX_WORKERS = int(os.environ.get('V4C_WORKERS', '8'))
TRACES_PATH = os.environ.get('V4C_TRACES_PATH', 'output/v4c_think_matres_test.traces.jsonl')
DOT_DIR = os.environ.get('V4C_DOT_DIR', 'output/v4c_think_matres_test_dot')
DOT_FILE = os.path.join(DOT_DIR, os.environ.get('V4C_DOT_FILE_NAME', 'matres_v4c_think.json'))
DOC_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'

YESNO_RE = re.compile(r'\b(yes|no)\b', re.IGNORECASE)


def parse_yesno_after_think(text):
    """Extract yes/no from tail after </Think> (or </think>)."""
    if not text:
        return None
    tail = text
    for tag in ('</Think>', '</think>'):
        if tag in tail:
            tail = tail.rsplit(tag, 1)[-1]
            break
    tail = tail.strip()
    # First non-empty token
    first = tail.split('\n', 1)[0].strip()
    m = YESNO_RE.search(first)
    if m:
        return m.group(1).lower()
    m = YESNO_RE.search(tail)
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
    choice = resp.choices[0]
    finish = getattr(choice, 'finish_reason', None)
    comp = resp.usage.completion_tokens if resp.usage else 0
    return choice.message.content or '', finish, comp


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
    q2 = answers.get('Q2'); q3 = answers.get('Q3'); q4 = answers.get('Q4')
    if q2 == 'yes': return 'EQUAL'
    if q2 != 'no':  return None
    if q3 == 'yes': return 'BEFORE'
    if q3 != 'no':  return None
    if q4 == 'yes': return 'AFTER'
    if q4 == 'no':  return 'VAGUE'
    return None


def load_doc_index():
    docs = {}
    for fn in sorted(os.listdir(DOC_FOLDER)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(DOC_FOLDER, fn)) as f:
            d = json.load(f)
        docs[fn] = {
            'tokens': d['tokens'],
            'mentions': {m['m_id']: m for m in d.get('allMentions', [])},
        }
    return docs


def find_doc_index():
    candidates = ['data/MATRES/_in_OmniTemp_format/test', DOC_FOLDER]
    docs = {}
    for c in candidates:
        if not os.path.isdir(c):
            continue
        for fn in sorted(os.listdir(c)):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(c, fn)) as f:
                d = json.load(f)
            if fn not in docs:
                docs[fn] = {
                    'tokens': d['tokens'],
                    'mentions': {m['m_id']: m for m in d.get('allMentions', [])},
                }
    return docs


def work_one(client, p, docs):
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
    raw = {}
    finish = {}
    comp_toks = {}
    messages = []
    t_start = time.time()

    def turn(qid, user_text):
        messages.append({'role': 'user', 'content': user_text})
        out, fin, ct = call_with_retry(client, messages)
        raw[qid] = out
        finish[qid] = fin
        comp_toks[qid] = ct
        a = parse_yesno_after_think(out)
        answers[qid] = a
        # feed full response back as next-turn context
        messages.append({'role': 'assistant', 'content': out})
        return a

    a1 = turn('Q1', construct_q1(marked_doc, e1_ref, e2_ref))
    if a1 is None:
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start, error='Q1 unparseable')
    in_event = (a1 == 'yes')

    a2 = turn('Q2', construct_q('Q2', e1_ref, e2_ref, in_event))
    if a2 is None:
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start, error='Q2 unparseable')
    if a2 == 'yes':
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start)

    a3 = turn('Q3', construct_q('Q3', e1_ref, e2_ref, in_event))
    if a3 is None:
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start, error='Q3 unparseable')
    if a3 == 'yes':
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start)

    a4 = turn('Q4', construct_q('Q4', e1_ref, e2_ref, in_event))
    if a4 is None:
        return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start, error='Q4 unparseable')
    return _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start)


def _trace(p, m1, m2, answers, raw, finish, comp_toks, t_start, error=None):
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
        'raw_responses': raw,
        'finish_reasons': finish,
        'completion_tokens': comp_toks,
        'predicted_label': label,
        'error': error,
        'turns_used': sum(1 for v in answers.values() if v in ('yes', 'no')),
        'wall_secs': round(time.time() - t_start, 1),
    }


def main():
    if not ENDPOINT_NAME:
        sys.exit('ERROR: set V4C_ENDPOINT_NAME')

    os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
    os.makedirs(DOT_DIR, exist_ok=True)

    pairs = load_test_pairs()
    print(f'loaded {len(pairs)} test pairs', flush=True)

    docs = find_doc_index()
    print(f'loaded doc index: {len(docs)} files', flush=True)

    done = set()
    traces = []
    if os.path.exists(TRACES_PATH):
        with open(TRACES_PATH) as f:
            for line in f:
                t = json.loads(line)
                done.add((t['doc_id'], t['e1_id'], t['e2_id']))
                traces.append(t)
        print(f'resuming: {len(done)} done', flush=True)

    remaining = [p for p in pairs if (p['doc_id'], p['e1_id'], p['e2_id']) not in done]
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
                        'gold_label': p['gold_label'], 'predicted_label': None,
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
    print(f'\nwall-clock: {wall/60:.1f} min')

    n = len(traces)
    n_unparse = sum(1 for t in traces if t.get('predicted_label') is None)
    pred_dist = Counter((t.get('predicted_label') or 'UNPARSEABLE') for t in traces)
    gold_dist = Counter(t['gold_label'] for t in traces)
    turn_dist = Counter(t.get('turns_used', 0) for t in traces)
    q1yes = sum(1 for t in traces if t.get('q1_in_event'))

    # Token budget diagnostics
    total_completion = sum(sum((t.get('completion_tokens') or {}).values()) for t in traces)
    truncated_turns = 0
    for t in traces:
        for q, fr in (t.get('finish_reasons') or {}).items():
            if fr == 'length':
                truncated_turns += 1

    print('\n======== diagnostics ========')
    print(f'total processed:                  {n}')
    print(f'  unparseable (no commit):        {n_unparse}')
    print(f'  Q1=Yes (in-event branch):       {q1yes} ({100*q1yes/max(1,n):.1f}%)')
    print(f'  turns-used dist:                {dict(sorted(turn_dist.items()))}')
    print(f'  gold dist:                      {dict(gold_dist)}')
    print(f'  pred dist:                      {dict(pred_dist)}')
    print(f'  total completion tokens:        {total_completion:,}')
    print(f'  turns truncated by max_tokens:  {truncated_turns}')

    per_doc = {}
    for t in traces:
        per_doc.setdefault(t['doc_id'], [])
        pred = t.get('predicted_label')
        if pred is None:
            continue
        e1 = str(t['e1_trigger']).replace('"', '')
        e2 = str(t['e2_trigger']).replace('"', '')
        per_doc[t['doc_id']].append(
            f'"{e1}({t["e1_id"]})" -- "{e2}({t["e2_id"]})" [rel={pred.lower()}];'
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
