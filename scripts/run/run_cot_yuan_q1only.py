"""Q1-only runner for MATRES training set — inspection data, NOT predictions.

For each pair, asks only Yuan Q1 ("are these referring to the same event?")
and saves the raw model response (including <think>…</think> reasoning from
DeepSeek R1) plus the parsed yes/no/uncertain.

No Q2/Q3/Q4, no branching, no label prediction. This is a sample of the
teacher's same-event reasoning, for inspection of the Q1 failure mode we
identified in the 49-pair Branch-A analysis.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_cot_yuan_q1only.py \\
        --api together --model_name deepseek-ai/DeepSeek-R1 \\
        --output_folder output/matres_train_q1only_pilot20 \\
        --limit 20 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    mark_target_pair_in_doc, ref, q_same_event, parse_yes_no,
)
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel, GPTModel


MATRES_TRAIN_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'

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


def build_q1_prompt(doc_tokens, m1, m2):
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'],
        m1['m_id'], m2['m_id'],
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])
    return q_same_event(doc_text, e1, e2)


def _worker(task, api, model_name):
    doc_id, m1, m2, gold = task
    llm = _thread_llm(api, model_name)
    llm.clear()
    prompt = build_q1_prompt(_doc_cache[doc_id]['tokens'], m1, m2)
    raw = llm.run_model_chat(prompt)
    parsed = parse_yes_no(raw)
    return {
        'doc_id': doc_id,
        'm1_id': m1['m_id'],
        'm2_id': m2['m_id'],
        't1': m1['tokens'],
        't2': m2['tokens'],
        'gold_label': gold,
        'q1_prompt': prompt,
        'q1_raw_response': raw,
        'q1_parsed': parsed,
    }


_doc_cache: dict = {}


def build_tasks(limit):
    files = sorted(f for f in os.listdir(MATRES_TRAIN_FOLDER)
                   if f.endswith('.json') or f.endswith('.jsonl'))
    tasks = []
    for file in files:
        data = open_input_file(f'{MATRES_TRAIN_FOLDER}/{file}')
        _doc_cache[file] = data
        ment_dict = {m['m_id']: m for m in data['allMentions']}
        for pair in data['allPairs']:
            m1 = ment_dict[pair['_firstId']]
            m2 = ment_dict[pair['_secondId']]
            tasks.append((file, m1, m2, pair.get('_relation')))
            if limit is not None and len(tasks) >= limit:
                return tasks
    return tasks


def main(api, model_name, output_file, workers, limit):
    tasks = build_tasks(limit)
    print(f"Loaded {len(tasks)} pairs from {MATRES_TRAIN_FOLDER}", file=sys.stderr)

    # Stdout redirect, done ONCE globally (not per-worker) to avoid the
    # thread-unsafe redirect_stdout pattern from the original runner.
    # tqdm + error prints go to stderr so they stay visible.
    if workers > 1:
        sys.stdout = open(os.devnull, 'w')

    lock = threading.Lock()
    n_written = 0
    with open(output_file, 'w') as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, api, model_name) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stderr):
                try:
                    rec = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}", file=sys.stderr)
                    continue
                with lock:
                    out.write(json.dumps(rec) + '\n')
                    out.flush()
                    n_written += 1

    print(f"Wrote {n_written}/{len(tasks)} records to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True,
                        help="together | gpt. gemini is not supported (no chat mode).")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on total pairs (for pilot runs).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.api == "gemini":
        raise SystemExit("Gemini is not supported: GeminiModel has no chat mode.")
    if args.api not in ("together", "gpt"):
        raise SystemExit(f"Unknown --api value: {args.api!r}. Expected 'together' or 'gpt'.")

    if os.path.exists(args.output_folder):
        print("Output folder already exists. Exiting to avoid overwrite.", file=sys.stderr)
        sys.exit(0)
    os.makedirs(args.output_folder)

    short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    output_file = f"{args.output_folder}/matres_train_q1only_{short}.jsonl"

    start = time.time()
    main(api=args.api, model_name=args.model_name,
         output_file=output_file, workers=args.workers, limit=args.limit)
    print(f"Took {time.time() - start:.1f}s", file=sys.stderr)
