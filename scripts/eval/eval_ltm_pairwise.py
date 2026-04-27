"""Eval LTM Step 1 (pairwise prompt → BEFORE/AFTER/EQUAL/VAGUE) on MATRES test.

For each MATRES test pair, send build_pairwise_prompt(doc, pair) and parse the
model's single-word response.

Usage (from GlobalZeroShotTRE/):
    LTM_ENDPOINT_NAME=<hashed-name> PYTHONPATH=. .venv/bin/python scripts/eval/eval_ltm_pairwise.py
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

from scripts.eval.eval_sft_student import _is_retryable
from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation


ENDPOINT_NAME = os.environ.get('LTM_ENDPOINT_NAME', '')
MAX_TOKENS = int(os.environ.get('LTM_MAX_TOKENS', '16'))
MAX_WORKERS = int(os.environ.get('LTM_WORKERS', '8'))
TEST_FOLDER = 'data/MATRES/_in_OmniTemp_format/test'
TRACES_PATH = os.environ.get('LTM_TRACES_PATH', 'output/ltm_step1_matres_test.traces.jsonl')
DOT_DIR = os.environ.get('LTM_DOT_DIR', 'output/ltm_step1_matres_test_dot')
DOT_FILE = os.path.join(DOT_DIR, os.environ.get('LTM_DOT_FILE_NAME', 'matres_ltm_step1.json'))


def mark_events_in_text(tokens, all_mentions):
    tokens = list(tokens)
    for mention in all_mentions:
        first = mention['tokens_ids'][0]
        last = mention['tokens_ids'][-1]
        tokens[first] = f'<{tokens[first]}'
        tokens[last] = f'{tokens[last]}(ei{mention["m_id"]})>'
    return ' '.join(tokens)


def build_pairwise_prompt(doc, pair):
    text = mark_events_in_text(doc['tokens'], doc['allMentions'])
    mention_text = {str(m['m_id']): str(m['tokens']) for m in doc['allMentions']}
    first_id = pair['_firstId']
    second_id = pair['_secondId']
    first = mention_text.get(str(first_id), str(first_id))
    second = mention_text.get(str(second_id), str(second_id))
    return (
        "\nGiven the text below where events are marked with <eventName(identifier)>,"
        " for the specified pair of events below, determine the temporal relationships "
        "(BEFORE, AFTER, EQUAL, VAGUE) between them.\n\nText -\n"
        f"{text}\n\n"
        "Pair -\n"
        f"{first}(ei{first_id}) -- {second}(ei{second_id})\n\n"
        "Answer -\n"
    )


REL_RE = re.compile(r'\b(before|after|equal|vague)\b', re.IGNORECASE)


def parse_relation(text):
    m = REL_RE.search(text or '')
    return m.group(1).upper() if m else None


def call_with_retry(client, prompt, max_attempts=5):
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=ENDPOINT_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ''
        except Exception as e:
            last = e
            if _is_retryable(e) and attempt < max_attempts:
                wait = min(60, 2 ** attempt)
                print(f'[retry {attempt}/{max_attempts}] {type(e).__name__}: {e} — sleep {wait}s',
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise
    raise last


def load_test_pairs():
    """Yield (doc_id, doc, pair) tuples for every MATRES test pair."""
    pairs = []
    docs = {}
    for fn in sorted(os.listdir(TEST_FOLDER)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(TEST_FOLDER, fn)) as f:
            d = json.load(f)
        docs[fn] = d
        for p in d.get('allPairs', []):
            pairs.append((fn, p))
    return docs, pairs


def work_pair(client, doc, pair, doc_id):
    prompt = build_pairwise_prompt(doc, pair)
    t0 = time.time()
    try:
        out = call_with_retry(client, prompt)
        pred = parse_relation(out)
    except Exception as e:
        out = f'__ERROR__: {type(e).__name__}: {e}'
        pred = None
    mention_text = {str(m['m_id']): str(m['tokens']) for m in doc['allMentions']}
    return {
        'doc_id': doc_id,
        'e1_id': str(pair['_firstId']),
        'e2_id': str(pair['_secondId']),
        'e1_trigger': mention_text.get(str(pair['_firstId']), ''),
        'e2_trigger': mention_text.get(str(pair['_secondId']), ''),
        'gold_label': pair['_relation'],
        'predicted_label': pred,
        'raw_output': out,
        'wall_secs': round(time.time() - t0, 1),
    }


def main():
    if not ENDPOINT_NAME:
        sys.exit('ERROR: set LTM_ENDPOINT_NAME')

    os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
    os.makedirs(DOT_DIR, exist_ok=True)

    docs, pairs = load_test_pairs()
    print(f'loaded {len(docs)} test docs, {len(pairs)} pairs', flush=True)

    done = set()
    traces = []
    if os.path.exists(TRACES_PATH):
        with open(TRACES_PATH) as f:
            for line in f:
                t = json.loads(line)
                done.add((t['doc_id'], t['e1_id'], t['e2_id']))
                traces.append(t)
        print(f'resuming: {len(done)} pairs already traced', flush=True)

    remaining = [(did, p) for did, p in pairs
                 if (did, str(p['_firstId']), str(p['_secondId'])) not in done]
    print(f'remaining: {len(remaining)} pairs via {ENDPOINT_NAME}', flush=True)

    client = Together()
    start = time.time()
    if remaining:
        with open(TRACES_PATH, 'a') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(work_pair, client, docs[did], p, did): (did, p) for did, p in remaining}
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

    wall = time.time() - start
    print(f'\nwall-clock: {wall/60:.1f} min')

    n = len(traces)
    n_unparse = sum(1 for t in traces if t.get('predicted_label') is None)
    pred_dist = Counter((t.get('predicted_label') or 'UNPARSEABLE') for t in traces)
    gold_dist = Counter(t['gold_label'] for t in traces)
    print('\n======== diagnostics ========')
    print(f'total processed:       {n}')
    print(f'unparseable:           {n_unparse}')
    print(f'gold dist:             {dict(gold_dist)}')
    print(f'pred dist:             {dict(pred_dist)}')

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
