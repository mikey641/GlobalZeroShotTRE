"""Pilot: gold-conditioned 'why' prompts on DeepSeek-R1.

Single-turn, stateless. For each selected pair we ask two independent 'why'
questions (Q_pos = gold framing, Q_neg = negated-complement framing) and save
the full responses (including <think>...</think>) for eyeballing.

Selection is done up front in __main__ by reading the continuation traces and
picking 4 BEFORE + 4 AFTER pairs with a mix of teacher-correct/wrong and
Branch A / Branch B.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc, ref
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel


# Same retry policy as run_cot_yuan_continue.py.
_RETRY_SUBSTRINGS = (
    'RateLimitError', 'ReadTimeout', 'InternalServerError',
    'RemoteProtocolError', 'APIConnectionError', 'ServiceUnavailableError',
    'APITimeoutError', 'ConnectError',
)
MAX_RETRIES = 6


def _is_retryable(err):
    n = type(err).__name__
    if any(s in n for s in _RETRY_SUBSTRINGS):
        return True
    sc = getattr(err, 'status_code', None)
    if sc in (408, 429, 500, 502, 503, 504):
        return True
    return False


def _run_with_retry(llm, prompt):
    """Single-turn call with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return llm.run_model(prompt)
        except Exception as e:
            if not _is_retryable(e) or attempt == MAX_RETRIES:
                raise
            sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
            print(f"[retry {attempt+1}/{MAX_RETRIES}] {type(e).__name__}: "
                  f"sleeping {sleep_s:.1f}s", file=sys.stderr, flush=True)
            time.sleep(sleep_s)


_thread_local = threading.local()


def _thread_llm(model_name):
    if not hasattr(_thread_local, 'llm'):
        _thread_local.llm = TogetherModel(model_name)
    return _thread_local.llm


def q_why_pos(doc_text, e1_ref, e2_ref, gold):
    relation = 'before' if gold == 'BEFORE' else 'after'
    return (
        f"Given the following document:\n\n{doc_text}\n\n"
        f"Explain why {e1_ref} is {relation} {e2_ref}. "
        f"Keep the answer short and concise."
    )


def q_why_neg(doc_text, e1_ref, e2_ref, gold):
    # The complement of BEFORE is AFTER (and vice versa); we negate the complement.
    complement = 'after' if gold == 'BEFORE' else 'before'
    return (
        f"Given the following document:\n\n{doc_text}\n\n"
        f"Explain why {e1_ref} is not {complement} {e2_ref}. "
        f"Keep the answer short and concise."
    )


def _worker(task, model_name, suppress_stdout):
    row, doc_tokens, m1, m2 = task
    llm = _thread_llm(model_name)
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'],
        m1['m_id'], m2['m_id'],
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])
    gold = row['gold_label']

    q_pos = q_why_pos(doc_text, e1, e2, gold)
    q_neg = q_why_neg(doc_text, e1, e2, gold)

    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            r_pos = _run_with_retry(llm, q_pos)
            r_neg = _run_with_retry(llm, q_neg)
    else:
        r_pos = _run_with_retry(llm, q_pos)
        r_neg = _run_with_retry(llm, q_neg)

    return {
        'doc_id': row['doc_id'],
        'e1_id': m1['m_id'],
        'e2_id': m2['m_id'],
        'e1_trigger': m1['tokens'],
        'e2_trigger': m2['tokens'],
        'gold_label': gold,
        'original_predicted_label': row['predicted_label'],
        'q_pos_question': q_pos,
        'q_pos_response': r_pos,
        'q_neg_question': q_neg,
        'q_neg_response': r_neg,
    }


def pick_pairs(traces_path):
    """Pick 4 BEFORE + 4 AFTER pairs with correct/wrong x Branch A/B coverage."""
    rows = [json.loads(l) for l in open(traces_path)]

    def branch(r):
        return r['turns'][0]['parsed']

    def find(gold, correct, branch_want):
        for r in rows:
            if r['gold_label'] != gold:
                continue
            hit = r['predicted_label'] == gold
            if hit != correct:
                continue
            if branch(r) != branch_want:
                continue
            return r
        return None

    selected = []
    for gold in ('BEFORE', 'AFTER'):
        for correct in (True, False):
            for branch_want in ('yes', 'no'):
                r = find(gold, correct, branch_want)
                if r is not None:
                    selected.append(r)
    assert len(selected) == 8, f"expected 8, got {len(selected)}"
    return selected


def attach_mentions(rows, matres_folder):
    """For each selected trace row, load its doc to get tokens + mention objects."""
    tasks = []
    for r in rows:
        data = open_input_file(f"{matres_folder}/{r['doc_id']}")
        tokens = data['tokens']
        ment_dict = {m['m_id']: m for m in data['allMentions']}
        m1 = ment_dict[r['e1_id']]
        m2 = ment_dict[r['e2_id']]
        tasks.append((r, tokens, m1, m2))
    return tasks


def pretty_print(results):
    for i, r in enumerate(results, 1):
        print("=" * 100)
        print(f"[{i}/8] doc={r['doc_id']}  e1={r['e1_id']}({r['e1_trigger']})  "
              f"e2={r['e2_id']}({r['e2_trigger']})  "
              f"gold={r['gold_label']}  orig_pred={r['original_predicted_label']}")
        print("-" * 100)
        print("[Q_POS PROMPT]")
        print(r['q_pos_question'])
        print()
        print("[Q_POS RESPONSE]")
        print(r['q_pos_response'])
        print()
        print("-" * 100)
        print("[Q_NEG PROMPT]")
        print(r['q_neg_question'])
        print()
        print("[Q_NEG RESPONSE]")
        print(r['q_neg_response'])
        print()


def main(model_name, traces_file, output_folder, matres_folder, workers):
    rows = pick_pairs(traces_file)
    print(f"Picked {len(rows)} pairs. "
          f"gold: {dict(Counter(r['gold_label'] for r in rows))}  "
          f"original_pred: {dict(Counter(r['predicted_label'] for r in rows))}  "
          f"Q1: {dict(Counter(r['turns'][0]['parsed'] for r in rows))}",
          file=sys.stderr)

    tasks = attach_mentions(rows, matres_folder)
    out_path = f"{output_folder}/pilot.traces.jsonl"
    suppress = workers > 1
    lock = threading.Lock()
    results = []

    with open(out_path, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, model_name, suppress) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stderr):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}", file=sys.stderr)
                    continue
                results.append(rec)
                with lock:
                    tf.write(json.dumps(rec) + '\n')
                    tf.flush()

    # Re-order results to match the original selection order for readable printing.
    by_key = {(r['doc_id'], r['e1_id'], r['e2_id']): r for r in results}
    ordered = []
    for row in rows:
        k = (row['doc_id'], row['e1_id'], row['e2_id'])
        if k in by_key:
            ordered.append(by_key[k])

    pretty_print(ordered)
    print(f"Saved {len(results)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1")
    parser.add_argument(
        "--traces_file",
        default="output/matres_train_continue_full/"
                "matres_train_continue_DeepSeek-R1.traces.jsonl",
    )
    parser.add_argument("--output_folder", default="output/matres_pilot_why_bf_af")
    parser.add_argument("--matres_folder",
                        default="data/MATRES/_in_OmniTemp_format/train")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        print("Output folder already exists. Exiting to avoid overwrite.",
              file=sys.stderr)
        sys.exit(0)
    os.makedirs(args.output_folder)

    start = time.time()
    main(model_name=args.model_name,
         traces_file=args.traces_file,
         output_folder=args.output_folder,
         matres_folder=args.matres_folder,
         workers=args.workers)
    print(f"Took {time.time() - start:.1f}s", file=sys.stderr)
