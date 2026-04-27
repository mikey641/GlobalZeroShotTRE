"""Gemini-2.5-Pro Yuan CoT runner over the R1-mismatch subset of MATRES train.

Reuses the existing run_cot_yuan_gemini.py multi-turn chat protocol, but:
  - Iterates over MATRES TRAIN pairs where R1's predicted_label != gold_label
  - Trace output schema matches matres_train_continue_DeepSeek-R1.traces.jsonl
    (so the resulting file slots into the v4d data-build pipeline)
  - thinking_budget = 8192 (per spec)
  - On unparseable Q answer, retry once; if still unparseable, skip the pair

NO gold conditioning. Forward-pass only — see if Gemini independently reaches
gold more often than R1 did on these specific pairs.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_cot_yuan_gemini_for_r1mismatch.py \\
        --output_file output/matres_train_continue_gemini25pro_calib50.jsonl \\
        --max_pairs 50 --stratified --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    mark_target_pair_in_doc, ref,
    q_same_event,
    q_simultaneous_same_event, q_simultaneous,
    q_before_same_event, q_after_same_event,
    q_before, q_after,
    parse_yes_no,
)
from scripts.utils.io_utils import open_input_file


R1_TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'
TRAIN_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'
THINKING_BUDGET = 8192

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 6

_thread_local = threading.local()


def _send_with_retry(chat, question):
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat.send_message(question)
        except genai_errors.APIError as e:
            code = getattr(e, 'code', None)
            if code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            last = e
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            last = e
        sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
        print(f'[retry {attempt+1}/{MAX_RETRIES}] {type(last).__name__}: sleeping {sleep_s:.1f}s',
              file=sys.stderr, flush=True)
        time.sleep(sleep_s)


def _thread_client(model_name):
    if not hasattr(_thread_local, 'client'):
        project = os.environ['GOOGLE_CLOUD_PROJECT']
        location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
        _thread_local.client = genai.Client(vertexai=True, project=project, location=location)
        _thread_local.model_name = model_name
    return _thread_local.client


def _new_chat(client, model_name):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=THINKING_BUDGET,
            ),
        ),
    )


def _extract(response):
    """Return (thought_text, answer_text, usage_dict) from Gemini response."""
    thoughts, answer = [], []
    cands = getattr(response, 'candidates', None) or []
    if cands:
        parts = getattr(cands[0].content, 'parts', None) or []
        for p in parts:
            t = getattr(p, 'text', '') or ''
            if getattr(p, 'thought', False):
                thoughts.append(t)
            else:
                answer.append(t)
    if not answer and not thoughts:
        answer.append(getattr(response, 'text', '') or '')
    usage = getattr(response, 'usage_metadata', None)
    usage_d = {}
    if usage is not None:
        for k in ('prompt_token_count', 'candidates_token_count', 'total_token_count',
                  'thoughts_token_count'):
            v = getattr(usage, k, None)
            if v is not None:
                usage_d[k] = int(v)
    return ''.join(thoughts), ''.join(answer), usage_d


class UnparseableError(Exception):
    pass


def _ask(chat, question, trace, usage_acc):
    """Send one turn. Retry once on unparseable. Raise UnparseableError if still bad."""
    resp = _send_with_retry(chat, question)
    thought, answer, usage = _extract(resp)
    parsed = parse_yes_no(answer)
    for k, v in usage.items():
        usage_acc[k] = usage_acc.get(k, 0) + v

    if parsed in ('yes', 'no'):
        trace.append({'question': question, 'thought': thought,
                      'response': answer, 'parsed': parsed})
        return parsed

    # Unparseable: one retry with a clarification turn
    print(f'[unparseable] retrying once: parsed={parsed!r} answer={answer[:80]!r}',
          file=sys.stderr, flush=True)
    clar_q = 'Please answer with just "yes" or "no".'
    resp2 = _send_with_retry(chat, clar_q)
    thought2, answer2, usage2 = _extract(resp2)
    parsed2 = parse_yes_no(answer2)
    for k, v in usage2.items():
        usage_acc[k] = usage_acc.get(k, 0) + v

    if parsed2 in ('yes', 'no'):
        trace.append({'question': question, 'thought': thought,
                      'response': answer, 'parsed': parsed, 'retry_clarification': True,
                      'retry_response': answer2, 'retry_parsed': parsed2})
        return parsed2

    raise UnparseableError(f'Q={question[:80]!r} answer1={answer[:80]!r} answer2={answer2[:80]!r}')


def run_cot_pair(chat, doc_tokens, m1, m2, usage_acc):
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'], m1['m_id'], m2['m_id']
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])
    trace = []
    same = _ask(chat, q_same_event(doc_text, e1, e2), trace, usage_acc)

    if same == 'yes':
        if _ask(chat, q_simultaneous_same_event(e1, e2), trace, usage_acc) == 'yes':
            return 'EQUAL', trace
        if _ask(chat, q_before_same_event(e1, e2), trace, usage_acc) == 'yes':
            return 'BEFORE', trace
        if _ask(chat, q_after_same_event(e1, e2), trace, usage_acc) == 'yes':
            return 'AFTER', trace
        return 'VAGUE', trace

    if _ask(chat, q_simultaneous(e1, e2), trace, usage_acc) == 'yes':
        return 'EQUAL', trace
    if _ask(chat, q_before(e1, e2), trace, usage_acc) == 'yes':
        return 'BEFORE', trace
    if _ask(chat, q_after(e1, e2), trace, usage_acc) == 'yes':
        return 'AFTER', trace
    return 'VAGUE', trace


def load_r1_mismatch_pairs():
    """Yields list of {doc_id, e1_id, e2_id, gold_label, r1_predicted_label}."""
    rows = []
    with open(R1_TRACES) as f:
        for line in f:
            t = json.loads(line)
            if t['predicted_label'] != t['gold_label']:
                rows.append({
                    'doc_id': t['doc_id'], 'e1_id': t['e1_id'], 'e2_id': t['e2_id'],
                    'e1_trigger': t['e1_trigger'], 'e2_trigger': t['e2_trigger'],
                    'gold_label': t['gold_label'],
                    'r1_predicted_label': t['predicted_label'],
                })
    return rows


def stratified_sample(rows, n, seed=0):
    """Stratified by gold_label, proportional to subset distribution."""
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['gold_label']].append(r)
    total = len(rows)
    quotas = {c: round(n * len(rs) / total) for c, rs in by_class.items()}
    diff = n - sum(quotas.values())
    if diff != 0:
        # adjust largest class
        largest = max(quotas, key=quotas.get)
        quotas[largest] += diff
    rng = random.Random(seed)
    picked = []
    for c, q in quotas.items():
        rs = list(by_class[c])
        rng.shuffle(rs)
        picked.extend(rs[:q])
    rng.shuffle(picked)
    return picked


def load_doc_index():
    docs = {}
    for fn in sorted(os.listdir(TRAIN_FOLDER)):
        if not fn.endswith('.json'):
            continue
        d = open_input_file(os.path.join(TRAIN_FOLDER, fn))
        docs[fn] = d
    return docs


def _worker(task, model_name):
    pair_meta, doc = task
    client = _thread_client(model_name)
    chat = _new_chat(client, model_name)
    ment_dict = {m['m_id']: m for m in doc['allMentions']}
    m1 = ment_dict.get(str(pair_meta['e1_id']))
    m2 = ment_dict.get(str(pair_meta['e2_id']))
    if m1 is None or m2 is None:
        return pair_meta, None, None, {'missing_mention': True}
    usage_acc = {}
    try:
        label, trace = run_cot_pair(chat, doc['tokens'], m1, m2, usage_acc)
        return pair_meta, label, trace, usage_acc
    except UnparseableError as e:
        return pair_meta, None, None, {'unparseable': str(e), **usage_acc}


def main(model_name, output_file, max_pairs, workers, stratified):
    rows = load_r1_mismatch_pairs()
    print(f'R1-mismatch pairs total: {len(rows)}', file=sys.stderr)

    if stratified and max_pairs:
        sample = stratified_sample(rows, max_pairs)
        print(f'Stratified sample of {len(sample)}:', file=sys.stderr)
        for c, n in Counter(r['gold_label'] for r in sample).most_common():
            print(f'  {c}: {n}', file=sys.stderr)
    elif max_pairs:
        sample = rows[:max_pairs]
    else:
        sample = rows

    docs = load_doc_index()
    tasks = []
    for r in sample:
        d = docs.get(r['doc_id'])
        if d is None:
            print(f'  skip — missing doc {r["doc_id"]}', file=sys.stderr)
            continue
        tasks.append((r, d))

    n_match = 0
    n_unparse = 0
    n_skip = 0
    per_class_seen = Counter()
    per_class_match = Counter()
    total_usage = Counter()
    start = time.time()

    lock = threading.Lock()
    with open(output_file, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_worker, t, model_name) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), file=sys.stderr):
                try:
                    pair_meta, label, trace, usage = fut.result()
                except Exception as e:
                    print(f'task failed: {e!r}', file=sys.stderr)
                    n_skip += 1
                    continue
                for k, v in usage.items():
                    if isinstance(v, int):
                        total_usage[k] += v
                gold = pair_meta['gold_label']
                if label is None:
                    n_unparse += 1
                    with lock:
                        tf.write(json.dumps({
                            **pair_meta,
                            'predicted_label': None,
                            'turns': trace or [],
                            'error': usage.get('unparseable') or 'missing_mention',
                            'usage': {k: v for k, v in usage.items() if isinstance(v, int)},
                        }) + '\n')
                        tf.flush()
                    continue
                per_class_seen[gold] += 1
                if label == gold:
                    n_match += 1
                    per_class_match[gold] += 1
                with lock:
                    tf.write(json.dumps({
                        **pair_meta,
                        'predicted_label': label,
                        'turns': trace,
                        'usage': {k: v for k, v in usage.items() if isinstance(v, int)},
                    }) + '\n')
                    tf.flush()

    elapsed = time.time() - start
    n_scored = sum(per_class_seen.values())
    print('\n========= report =========', file=sys.stderr)
    print(f'wall: {elapsed/60:.1f} min', file=sys.stderr)
    print(f'attempted:    {len(tasks)}', file=sys.stderr)
    print(f'unparseable:  {n_unparse}', file=sys.stderr)
    print(f'errors/skip:  {n_skip}', file=sys.stderr)
    print(f'scored:       {n_scored}', file=sys.stderr)
    if n_scored:
        print(f'match rate:   {n_match}/{n_scored} = {100*n_match/n_scored:.1f}%', file=sys.stderr)
        for c in ('BEFORE', 'AFTER', 'EQUAL', 'VAGUE'):
            seen = per_class_seen.get(c, 0)
            if seen:
                print(f'  {c:7s}  {per_class_match.get(c, 0)}/{seen}  = {100*per_class_match.get(c,0)/seen:.1f}%',
                      file=sys.stderr)
    print(f'usage tokens: {dict(total_usage)}', file=sys.stderr)
    # Gemini-2.5-Pro pricing (Vertex AI, as of 2026-04): $1.25 / 1M input, $10 / 1M output
    # thinking tokens billed at output rate
    in_tok = total_usage.get('prompt_token_count', 0)
    out_tok = total_usage.get('candidates_token_count', 0)
    think_tok = total_usage.get('thoughts_token_count', 0)
    cost = in_tok / 1e6 * 1.25 + (out_tok + think_tok) / 1e6 * 10
    print(f'estimated cost (this run): ${cost:.2f}  (in={in_tok:,} out={out_tok:,} think={think_tok:,})',
          file=sys.stderr)
    if max_pairs and len(tasks) > 0:
        full = 5009
        scaled = cost * full / len(tasks)
        print(f'estimated cost (full {full}): ${scaled:.2f}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gemini-2.5-pro')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stratified', action='store_true',
                        help='Stratified sample by gold_label proportion')
    args = parser.parse_args()

    if not os.environ.get('GOOGLE_CLOUD_PROJECT'):
        raise SystemExit('GOOGLE_CLOUD_PROJECT env var required (Vertex AI + ADC)')

    if os.path.exists(args.output_file):
        print(f'output_file already exists: {args.output_file}. Exiting.', file=sys.stderr)
        sys.exit(0)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args.model_name, args.output_file, args.max_pairs, args.workers, args.stratified)
