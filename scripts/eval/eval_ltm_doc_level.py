"""Eval LTM-style model on MATRES test (document-level, all pairs in one assistant turn).

Sends one prompt per test document containing all pairs, parses the assistant
response as a list of `event(eiX) RELATION event(eiY)` lines, builds DOT, and
scores via the MATRES evaluator.

Prompt format matches src/matres_prompt.py:build_prompt.

Usage (from GlobalZeroShotTRE/):
    LTM_ENDPOINT_NAME=<hashed-name> PYTHONPATH=. .venv/bin/python scripts/eval/eval_ltm_doc_level.py
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
MAX_TOKENS = int(os.environ.get('LTM_MAX_TOKENS', '4096'))
MAX_WORKERS = int(os.environ.get('LTM_WORKERS', '4'))
TEST_FOLDER = os.environ.get('LTM_TEST_FOLDER', 'data/MATRES/_in_OmniTemp_format/test')
TRAIN_FOLDER_FALLBACK = 'data/MATRES/_in_OmniTemp_format/train'  # in case some pairs need lookup
TRACES_PATH = os.environ.get('LTM_TRACES_PATH', 'output/ltm_replay_matres_test.traces.jsonl')
DOT_DIR = os.environ.get('LTM_DOT_DIR', 'output/ltm_replay_matres_test_dot')
DOT_FILE = os.path.join(DOT_DIR, os.environ.get('LTM_DOT_FILE_NAME', 'matres_ltm_replay.json'))


# Mirrors src/matres_prompt.py:build_prompt
def mark_events_in_text(tokens, all_mentions):
    tokens = list(tokens)
    for mention in all_mentions:
        first = mention['tokens_ids'][0]
        last = mention['tokens_ids'][-1]
        tokens[first] = f'<{tokens[first]}'
        tokens[last] = f'{tokens[last]}(ei{mention["m_id"]})>'
    return ' '.join(tokens)


def build_prompt(doc):
    text = mark_events_in_text(doc['tokens'], doc['allMentions'])
    mention_text = {str(m['m_id']): str(m['tokens']) for m in doc['allMentions']}
    pair_lines = []
    for p in doc['allPairs']:
        first_id = p['_firstId']
        second_id = p['_secondId']
        first = mention_text.get(str(first_id), str(first_id))
        second = mention_text.get(str(second_id), str(second_id))
        pair_lines.append(f"{first}(ei{first_id}) -- {second}(ei{second_id})")
    return (
        "\nGiven the text below where events are marked with <eventName(identifier)>, "
        "for each pair of events below, determine the temporal relationships (BEFORE, AFTER, "
        "EQUAL, VAGUE) between them.\n\nText -\n"
        f"{text}\n\n"
        "Pairs -\n"
        + "\n".join(pair_lines)
        + "\n\n\nAnswer -\n"
    )


# Mirrors src/predict_pairs.py:_parse_global_answer
ANSWER_LINE_RE = re.compile(
    r"\(ei(\w+)\)\s+\b(before|after|equal|vague)\b.*?\(ei(\w+)\)"
    r"|\(ei(\w+)\).*?\(ei(\w+)\).*?\b(before|after|equal|vague)\b",
    re.IGNORECASE,
)


def parse_global_answer(text):
    """Return {(first_id, second_id): relation} from a global answer block."""
    result = {}
    text = text.replace('\\n', '\n')
    # Process line-by-line for cleaner matches
    for line in text.splitlines():
        m = ANSWER_LINE_RE.search(line)
        if not m:
            continue
        if m.group(1):
            first_id, relation, second_id = m.group(1), m.group(2).lower(), m.group(3)
        else:
            first_id, second_id, relation = m.group(4), m.group(5), m.group(6).lower()
        result[(str(first_id), str(second_id))] = relation
    return result


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
            choice = resp.choices[0]
            return choice.message.content or '', getattr(choice, 'finish_reason', None), \
                   resp.usage.completion_tokens if resp.usage else 0
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


def load_test_docs():
    docs = {}
    for fn in sorted(os.listdir(TEST_FOLDER)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(TEST_FOLDER, fn)) as f:
            d = json.load(f)
        docs[fn] = d
    return docs


def work_doc(client, doc_id, doc):
    prompt = build_prompt(doc)
    t0 = time.time()
    try:
        out, finish, comp = call_with_retry(client, prompt)
        parsed = parse_global_answer(out)
    except Exception as e:
        return {
            'doc_id': doc_id,
            'error': f'{type(e).__name__}: {e}',
            'predictions': {},
            'raw_output': '',
            'finish_reason': 'error',
            'completion_tokens': 0,
            'wall_secs': round(time.time() - t0, 1),
        }
    # Convert tuple keys to strings for JSON serialization
    pred_dict = {f'{a}|{b}': rel for (a, b), rel in parsed.items()}
    return {
        'doc_id': doc_id,
        'n_gold_pairs': len(doc.get('allPairs', [])),
        'n_pred_pairs': len(pred_dict),
        'predictions': pred_dict,
        'raw_output': out,
        'finish_reason': finish,
        'completion_tokens': comp,
        'wall_secs': round(time.time() - t0, 1),
    }


def main():
    if not ENDPOINT_NAME:
        sys.exit('ERROR: set LTM_ENDPOINT_NAME')

    os.makedirs(os.path.dirname(TRACES_PATH), exist_ok=True)
    os.makedirs(DOT_DIR, exist_ok=True)

    docs = load_test_docs()
    print(f'loaded {len(docs)} test docs', flush=True)
    total_pairs = sum(len(d.get('allPairs', [])) for d in docs.values())
    print(f'total gold pairs: {total_pairs}', flush=True)

    done = set()
    traces = []
    if os.path.exists(TRACES_PATH):
        with open(TRACES_PATH) as f:
            for line in f:
                t = json.loads(line)
                done.add(t['doc_id'])
                traces.append(t)
        print(f'resuming: {len(done)} docs already traced', flush=True)

    remaining = [(did, d) for did, d in docs.items() if did not in done]
    print(f'remaining: {len(remaining)} docs via {ENDPOINT_NAME}', flush=True)

    client = Together()
    start = time.time()
    if remaining:
        with open(TRACES_PATH, 'a') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {pool.submit(work_doc, client, did, d): did for did, d in remaining}
            for i, fut in enumerate(as_completed(futs), 1):
                t = fut.result()
                f.write(json.dumps(t) + '\n')
                f.flush()
                traces.append(t)
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                eta = (len(remaining) - i) / rate if rate else 0
                print(f'[{i}/{len(remaining)}]  rate={rate:.2f}docs/s  eta={eta/60:.1f}min  '
                      f'(doc={t["doc_id"]}: {t.get("n_pred_pairs", 0)}/{t.get("n_gold_pairs", 0)} pairs '
                      f'parsed, {t.get("completion_tokens", 0)} tok, {t.get("wall_secs", 0)}s)',
                      flush=True)

    wall = time.time() - start
    print(f'\nwall-clock: {wall/60:.1f} min')

    # Diagnostics
    n_docs = len(traces)
    truncated = sum(1 for t in traces if t.get('finish_reason') == 'length')
    total_pred = sum(t.get('n_pred_pairs', 0) for t in traces)
    total_gold = sum(t.get('n_gold_pairs', 0) for t in traces)
    pred_rel_dist = Counter()
    for t in traces:
        for rel in (t.get('predictions') or {}).values():
            pred_rel_dist[rel.upper()] += 1
    print('\n======== diagnostics ========')
    print(f'docs processed:                {n_docs}')
    print(f'truncated by max_tokens:       {truncated}')
    print(f'total gold pairs:              {total_gold}')
    print(f'total predicted pairs:         {total_pred}  (coverage {100*total_pred/max(1,total_gold):.1f}%)')
    print(f'predicted relation distribution: {dict(pred_rel_dist)}')

    # Build DOT
    per_doc = {}
    # Need triggers — load doc to get mention text
    for t in traces:
        doc_id = t['doc_id']
        doc = docs.get(doc_id)
        if doc is None:
            continue
        mention_text = {str(m['m_id']): str(m['tokens']).replace('"', '') for m in doc['allMentions']}
        edges = []
        for key, rel in (t.get('predictions') or {}).items():
            a, b = key.split('|', 1)
            ta = mention_text.get(a, a)
            tb = mention_text.get(b, b)
            edges.append(f'"{ta}({a})" -- "{tb}({b})" [rel={rel.lower()}];')
        per_doc[doc_id] = edges

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
