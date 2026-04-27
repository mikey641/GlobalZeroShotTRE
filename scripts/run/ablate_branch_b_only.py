"""Ablation: re-run Branch A pairs with Branch B style only (no Q1, no 'in that event').

Reads a traces.jsonl from a prior run_cot_yuan.py run, picks the pairs where
Q1 was 'yes', and re-queries them with just:
    Q_sim -> yes => EQUAL
    Q_before -> yes => BEFORE
    Q_after -> yes => AFTER
    else => VAGUE

Uses the same prompts (q_simultaneous, q_before, q_after), parser, and document
marking as the original runner.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    mark_target_pair_in_doc, ref,
    q_simultaneous, q_before, q_after,
    parse_yes_no,
)
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel, GPTModel


MATRES_TEST_FOLDER = 'data/MATRES/_in_OmniTemp_format/test'

_thread_local = threading.local()


def _thread_llm(api, model_name):
    if not hasattr(_thread_local, 'llm'):
        if api == 'together':
            _thread_local.llm = TogetherModel(model_name)
        elif api == 'gpt':
            _thread_local.llm = GPTModel(model_name)
        else:
            raise ValueError(f"Unsupported api: {api!r}")
    return _thread_local.llm


def run_branch_b(llm, doc_tokens, m1, m2):
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'],
        m1['m_id'], m2['m_id'],
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])

    trace = []

    def ask(q, include_doc=False):
        prompt = (f"Given the following document:\n\n{doc_text}\n\n" + q) if include_doc else q
        a = llm.run_model_chat(prompt)
        parsed = parse_yes_no(a)
        trace.append({'question': prompt, 'response': a, 'parsed': parsed})
        return parsed

    if ask(q_simultaneous(e1, e2), include_doc=True) == 'yes':
        return 'EQUAL', trace
    if ask(q_before(e1, e2)) == 'yes':
        return 'BEFORE', trace
    if ask(q_after(e1, e2)) == 'yes':
        return 'AFTER', trace
    return 'VAGUE', trace


def _worker(task, api, model_name, suppress_stdout):
    doc_id, e1_id, e2_id, doc_tokens, m1, m2, gold = task
    llm = _thread_llm(api, model_name)
    llm.clear()
    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            label, trace = run_branch_b(llm, doc_tokens, m1, m2)
    else:
        label, trace = run_branch_b(llm, doc_tokens, m1, m2)
    return doc_id, e1_id, e2_id, m1, m2, gold, label, trace


def build_tasks(traces_file):
    """Pull (doc_id, e1_id, e2_id, gold) for every trace whose Q1 was 'yes'."""
    targets = []
    with open(traces_file) as f:
        for line in f:
            r = json.loads(line)
            if r['turns'][0]['parsed'] != 'yes':
                continue
            targets.append((r['doc_id'], r['e1_id'], r['e2_id'], r['gold_label']))

    # Load each doc once, attach tokens + mentions to the tuples.
    doc_cache = {}
    tasks = []
    for doc_id, e1_id, e2_id, gold in targets:
        if doc_id not in doc_cache:
            doc_cache[doc_id] = open_input_file(f'{MATRES_TEST_FOLDER}/{doc_id}')
        d = doc_cache[doc_id]
        ment = {m['m_id']: m for m in d['allMentions']}
        tasks.append((doc_id, e1_id, e2_id, d['tokens'], ment[e1_id], ment[e2_id], gold))
    return tasks


def main(api, model_name, traces_in, traces_out, workers):
    tasks = build_tasks(traces_in)
    print(f"Loaded {len(tasks)} Branch-A pairs to re-run.")

    suppress = workers > 1
    results = []
    with open(traces_out, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, api, model_name, suppress) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    doc_id, e1_id, e2_id, m1, m2, gold, label, trace = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}")
                    continue
                results.append((doc_id, e1_id, e2_id, m1, m2, gold, label))
                tf.write(json.dumps({
                    'doc_id': doc_id,
                    'e1_id': e1_id,
                    'e2_id': e2_id,
                    'e1_trigger': m1['tokens'],
                    'e2_trigger': m2['tokens'],
                    'gold_label': gold,
                    'predicted_label': label,
                    'turns': trace,
                }) + '\n')
                tf.flush()
    print(f"Wrote {len(results)} results to {traces_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="together")
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--traces_in", required=True)
    parser.add_argument("--traces_out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    start = time.time()
    main(api=args.api, model_name=args.model_name,
         traces_in=args.traces_in, traces_out=args.traces_out,
         workers=args.workers)
    print(f"Took {time.time() - start:.1f}s")
