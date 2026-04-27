"""Yuan et al. (2023) zero-shot CoT runner for MATRES pairwise TRE.

Reuses GlobalZeroShotTRE data loading and emits the same DOT prediction format
read_pred_dot_file consumes, so existing eval scripts score it unchanged.

Scope: MATRES only (4 labels: BEFORE/AFTER/EQUAL/VAGUE).
Model support: Together, GPT. Gemini rejected — no chat mode.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_cot_yuan.py \\
        --api together --model_name <model> \\
        --output_folder output/matres_cot_yuan \\
        --repeat 1 --workers 8
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from scripts.utils.llms_definitions import TogetherModel, GPTModel


MATRES_TEST_FOLDER = 'data/MATRES/_in_OmniTemp_format/test'

_thread_local = threading.local()


def _thread_llm(api, model_name):
    """Return the current thread's LLM instance, creating on first use."""
    if not hasattr(_thread_local, 'llm'):
        if api == 'together':
            _thread_local.llm = TogetherModel(model_name)
        elif api == 'gpt':
            _thread_local.llm = GPTModel(model_name)
        else:
            raise ValueError(f"Unsupported api: {api!r}")
    return _thread_local.llm


def run_cot_pair(llm, doc_tokens, m1, m2):
    """Yuan CoT protocol for one pair. Returns (label, trace_list)."""
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'],
        m1['m_id'], m2['m_id'],
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])

    trace = []

    def ask(q):
        a = llm.run_model_chat(q)
        parsed = parse_yes_no(a)
        trace.append({'question': q, 'response': a, 'parsed': parsed})
        return parsed

    same = ask(q_same_event(doc_text, e1, e2))

    if same == 'yes':
        # Branch A: same event. Order: EQUAL → BEFORE → AFTER → VAGUE.
        if ask(q_simultaneous_same_event(e1, e2)) == 'yes':
            return 'EQUAL', trace
        if ask(q_before_same_event(e1, e2)) == 'yes':
            return 'BEFORE', trace
        if ask(q_after_same_event(e1, e2)) == 'yes':
            return 'AFTER', trace
        return 'VAGUE', trace

    # Branch B: different events. Same order as Branch A.
    if ask(q_simultaneous(e1, e2)) == 'yes':
        return 'EQUAL', trace
    if ask(q_before(e1, e2)) == 'yes':
        return 'BEFORE', trace
    if ask(q_after(e1, e2)) == 'yes':
        return 'AFTER', trace
    return 'VAGUE', trace


def build_dot(edges):
    lines = ["strict graph {"]
    for m1_name, m1_id, m2_name, m2_id, label in edges:
        lines.append(f'"{m1_name}({m1_id})" -- "{m2_name}({m2_id})" [rel={label.lower()}];')
    lines.append("}")
    return "\n".join(lines)


def _sort_pairs_by_doc_position(all_pairs, ment_dict):
    """Match run_zsl_tre.py default ordering: sort by first-event token position."""
    return sorted(
        all_pairs,
        key=lambda p: ment_dict[p['_firstId']]['tokens_ids'][0],
    )


def _worker(task, api, model_name, suppress_stdout):
    file, pair_idx, doc_tokens, m1, m2, gold = task
    llm = _thread_llm(api, model_name)
    llm.clear()
    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            label, trace = run_cot_pair(llm, doc_tokens, m1, m2)
    else:
        label, trace = run_cot_pair(llm, doc_tokens, m1, m2)
    return file, pair_idx, m1, m2, gold, label, trace


def main(api, model_name, output_file, traces_file,
         selected_file=None, max_pairs=None, workers=1):
    files = sorted(f for f in os.listdir(MATRES_TEST_FOLDER)
                   if f.endswith('.json') or f.endswith('.jsonl'))
    if selected_file is not None:
        files = [f for f in files if f == selected_file]

    tasks = []
    for file in files:
        data = open_input_file(f'{MATRES_TEST_FOLDER}/{file}')
        tokens = data['tokens']
        ment_dict = {m['m_id']: m for m in data['allMentions']}
        all_pairs = _sort_pairs_by_doc_position(data['allPairs'], ment_dict)
        for idx, pair in enumerate(all_pairs):
            m1 = ment_dict[pair['_firstId']]
            m2 = ment_dict[pair['_secondId']]
            tasks.append((file, idx, tokens, m1, m2, pair.get('_relation')))

    if max_pairs is not None:
        tasks = tasks[:max_pairs]

    # file -> {pair_idx -> (m1, m2, label)}
    file_results = {}
    trace_lock = threading.Lock()
    suppress = workers > 1

    with open(traces_file, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, api, model_name, suppress) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    file, pair_idx, m1, m2, gold, label, trace = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}")
                    continue
                file_results.setdefault(file, {})[pair_idx] = (m1, m2, label)
                with trace_lock:
                    tf.write(json.dumps({
                        'doc_id': file,
                        'e1_id': m1['m_id'],
                        'e2_id': m2['m_id'],
                        'e1_trigger': m1['tokens'],
                        'e2_trigger': m2['tokens'],
                        'gold_label': gold,
                        'predicted_label': label,
                        'turns': trace,
                    }) + '\n')
                    tf.flush()

    predictions = {}
    for file in files:
        if file not in file_results:
            continue
        edges = []
        for idx in sorted(file_results[file].keys()):
            m1, m2, label = file_results[file][idx]
            edges.append((m1['tokens'], m1['m_id'], m2['tokens'], m2['m_id'], label))
        predictions[file] = {"target": build_dot(edges)}

    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)


def _derive_short_name(model_name):
    return model_name.split('/')[-1] if '/' in model_name else model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True,
                        help="together | gpt. gemini is not supported (no chat mode).")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--selected_file", default=None)
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Hard cap on total pairs across all files (sanity checks)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent API calls (default: 1, serial)")
    args = parser.parse_args()

    if args.api == "gemini":
        raise SystemExit("Gemini is not supported: GeminiModel has no chat mode.")
    if args.api not in ("together", "gpt"):
        raise SystemExit(f"Unknown --api value: {args.api!r}. Expected 'together' or 'gpt'.")

    if os.path.exists(args.output_folder):
        print("Output folder already exists. Exiting to avoid overwrite.")
        sys.exit(0)
    os.makedirs(args.output_folder)

    short = _derive_short_name(args.model_name)

    for i in range(args.repeat):
        tag = f"matres_{short}_cot_yuan_{i}"
        output_file = f"{args.output_folder}/{tag}.json"
        traces_file = f"{args.output_folder}/{tag}.traces.jsonl"
        start = time.time()
        main(api=args.api, model_name=args.model_name,
             output_file=output_file, traces_file=traces_file,
             selected_file=args.selected_file, max_pairs=args.max_pairs,
             workers=args.workers)
        print(f"Run {i} took {time.time() - start:.1f}s")
