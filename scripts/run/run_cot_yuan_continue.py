"""Continuation runner: Q1 already answered per-pair, ask Q2/Q3/Q4 now.

Seeds each worker's chat history with the prior Q1 exchange (user=Q1 prompt,
assistant=Q1 raw response including any <think>…</think> block), then asks
Q2/Q3/Q4 branched on the saved q1_parsed value. Output trace format matches
run_cot_yuan.py byte-for-byte — the Q1 turn is reconstructed as the first
entry of `turns`, followed by the new Q2/Q3/Q4 turns from this run.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_cot_yuan_continue.py \\
        --api together --model_name deepseek-ai/DeepSeek-R1 \\
        --q1_traces_file output/matres_train_q1only_full/matres_train_q1only_DeepSeek-R1.jsonl \\
        --output_folder output/matres_train_continue_full \\
        --workers 8
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    ref,
    q_simultaneous_same_event, q_before_same_event, q_after_same_event,
    q_simultaneous, q_before, q_after,
    parse_yes_no,
)
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel, GPTModel


# Transient Together-API errors we've seen in prior runs: RateLimitError, ReadTimeout,
# InternalServerError, RemoteProtocolError. Match on substring rather than class to avoid
# importing the full together/httpx exception hierarchy.
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
    # HTTP status probe (together SDK wraps APIStatusError with .status_code)
    sc = getattr(err, 'status_code', None)
    if sc in (408, 429, 500, 502, 503, 504):
        return True
    return False


def _chat_with_retry(llm, question):
    """Wrap llm.run_model_chat with exponential backoff on transient errors.

    On retry we must NOT re-append to llm.messages — the original run_model_chat
    appends {user:question} on entry. If the API call raises, that append already
    happened and we must pop it before the retry, otherwise the message list grows
    duplicates on every retry.
    """
    for attempt in range(MAX_RETRIES + 1):
        msgs_before = len(llm.messages)
        try:
            return llm.run_model_chat(question)
        except Exception as e:
            # Roll back any partial state from the failed attempt.
            if len(llm.messages) > msgs_before:
                del llm.messages[msgs_before:]
            if not _is_retryable(e) or attempt == MAX_RETRIES:
                raise
            sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
            print(f"[retry {attempt+1}/{MAX_RETRIES}] {type(e).__name__}: "
                  f"sleeping {sleep_s:.1f}s", file=sys.stderr, flush=True)
            time.sleep(sleep_s)


DEFAULT_TRAIN_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'

_thread_local = threading.local()
_debug_consumed = threading.Event()  # True after we've printed debug for one task


def _thread_llm(api, model_name):
    if not hasattr(_thread_local, 'llm'):
        if api == 'together':
            _thread_local.llm = TogetherModel(model_name)
        elif api == 'gpt':
            _thread_local.llm = GPTModel(model_name)
        else:
            raise ValueError(f"Unsupported api: {api!r}")
    return _thread_local.llm


def _seed_q1_history(llm, q1_prompt, q1_raw_response):
    """Wipe state and plant the Q1 exchange as turn 1 of this worker's chat.

    Direct mutation of llm.messages is correct here: TogetherModel.run_model_chat
    (and GPTModel.run_model_chat) append {user:prompt} then call the API with the
    full self.messages list. So seeding with [user:Q1, assistant:Q1_response]
    results in the Q2 API request carrying [user:Q1, assistant:Q1_response, user:Q2]
    — exactly the continuation we want.
    """
    llm.clear()
    llm.messages.append({"role": "user", "content": q1_prompt})
    llm.messages.append({"role": "assistant", "content": q1_raw_response})


def _debug_print_history(llm, pair_key):
    print(f"\n[debug] messages before first Q2 call for pair {pair_key} "
          f"(len={len(llm.messages)}):", file=sys.stderr, flush=True)
    for i, m in enumerate(llm.messages):
        c = m['content']
        preview = c if len(c) <= 600 else c[:300] + f"  … [truncated {len(c)-600} chars] …  " + c[-300:]
        print(f"  [{i}] {m['role']}:\n{preview}\n", file=sys.stderr, flush=True)


def run_branch(llm, q1_parsed, e1_ref, e2_ref, debug_pair_key):
    """Ask Q2/Q3/Q4 according to the branch Q1 selected. Returns (label, new_turns)."""
    trace = []

    def ask(q):
        # On the very first ask across the whole run, if debug is armed, print messages.
        if debug_pair_key is not None and not trace and not _debug_consumed.is_set():
            _debug_consumed.set()
            _debug_print_history(llm, debug_pair_key)
        a = _chat_with_retry(llm, q)
        parsed = parse_yes_no(a)
        trace.append({'question': q, 'response': a, 'parsed': parsed})
        return parsed

    if q1_parsed == 'yes':
        if ask(q_simultaneous_same_event(e1_ref, e2_ref)) == 'yes':
            return 'EQUAL', trace
        if ask(q_before_same_event(e1_ref, e2_ref)) == 'yes':
            return 'BEFORE', trace
        if ask(q_after_same_event(e1_ref, e2_ref)) == 'yes':
            return 'AFTER', trace
        return 'VAGUE', trace

    # Anything that isn't an unambiguous 'yes' (including 'no' and 'uncertain')
    # takes Branch B — matches the original run_cot_yuan.py behavior.
    if ask(q_simultaneous(e1_ref, e2_ref)) == 'yes':
        return 'EQUAL', trace
    if ask(q_before(e1_ref, e2_ref)) == 'yes':
        return 'BEFORE', trace
    if ask(q_after(e1_ref, e2_ref)) == 'yes':
        return 'AFTER', trace
    return 'VAGUE', trace


def _worker(task, api, model_name, suppress_stdout, debug_enabled):
    file, pair_idx, m1, m2, gold, q1 = task
    llm = _thread_llm(api, model_name)
    _seed_q1_history(llm, q1['q1_prompt'], q1['q1_raw_response'])
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])
    debug_key = (file, m1['m_id'], m2['m_id']) if debug_enabled else None
    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            label, new_turns = run_branch(llm, q1['q1_parsed'], e1, e2, debug_key)
    else:
        label, new_turns = run_branch(llm, q1['q1_parsed'], e1, e2, debug_key)
    return file, pair_idx, m1, m2, gold, label, new_turns, q1


def load_q1_lookup(path):
    lookup = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            lookup[(r['doc_id'], r['m1_id'], r['m2_id'])] = {
                'q1_prompt': r['q1_prompt'],
                'q1_raw_response': r['q1_raw_response'],
                'q1_parsed': r['q1_parsed'],
            }
    return lookup


def build_tasks(folder, q1_lookup, max_pairs, smoke_balance):
    """Emit (file, pair_idx, m1, m2, gold, q1) tuples for pairs with Q1 data."""
    files = sorted(f for f in os.listdir(folder)
                   if f.endswith('.json') or f.endswith('.jsonl'))
    tasks = []
    skipped = 0
    for file in files:
        data = open_input_file(f'{folder}/{file}')
        ment_dict = {m['m_id']: m for m in data['allMentions']}
        for pair_idx, pair in enumerate(data['allPairs']):
            m1 = ment_dict[pair['_firstId']]
            m2 = ment_dict[pair['_secondId']]
            q1 = q1_lookup.get((file, m1['m_id'], m2['m_id']))
            if q1 is None:
                skipped += 1
                continue
            tasks.append((file, pair_idx, m1, m2, pair.get('_relation'), q1))

    if smoke_balance:
        half = smoke_balance // 2
        yes_tasks = [t for t in tasks if t[5]['q1_parsed'] == 'yes'][:half]
        no_tasks = [t for t in tasks if t[5]['q1_parsed'] == 'no'][:smoke_balance - half]
        tasks = yes_tasks + no_tasks
    elif max_pairs is not None:
        tasks = tasks[:max_pairs]

    return tasks, skipped


def build_dot(edges):
    lines = ["strict graph {"]
    for m1_name, m1_id, m2_name, m2_id, label in edges:
        lines.append(f'"{m1_name}({m1_id})" -- "{m2_name}({m2_id})" [rel={label.lower()}];')
    lines.append("}")
    return "\n".join(lines)


def main(api, model_name, q1_file, output_file, traces_file,
         matres_folder, max_pairs, workers, debug_first, smoke_balance):
    q1_lookup = load_q1_lookup(q1_file)
    print(f"Loaded {len(q1_lookup)} Q1 entries from {q1_file}", file=sys.stderr)

    tasks, skipped = build_tasks(matres_folder, q1_lookup, max_pairs, smoke_balance)
    yes_n = sum(1 for t in tasks if t[5]['q1_parsed'] == 'yes')
    no_n = sum(1 for t in tasks if t[5]['q1_parsed'] == 'no')
    unc_n = sum(1 for t in tasks if t[5]['q1_parsed'] == 'uncertain')
    print(f"Built {len(tasks)} tasks ({skipped} skipped — no Q1 data). "
          f"Q1 split: yes={yes_n} no={no_n} uncertain={unc_n}",
          file=sys.stderr)

    suppress = workers > 1
    file_results = {}
    lock = threading.Lock()

    with open(traces_file, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, api, model_name, suppress, debug_first)
                       for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stderr):
                try:
                    file, pair_idx, m1, m2, gold, label, new_turns, q1 = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}", file=sys.stderr)
                    continue
                # Reconstruct Q1 turn so the trace matches a fresh run_cot_yuan.py run.
                q1_turn = {
                    'question': q1['q1_prompt'],
                    'response': q1['q1_raw_response'],
                    'parsed': q1['q1_parsed'],
                }
                turns = [q1_turn] + new_turns
                file_results.setdefault(file, {})[pair_idx] = (m1, m2, label)
                with lock:
                    tf.write(json.dumps({
                        'doc_id': file,
                        'e1_id': m1['m_id'],
                        'e2_id': m2['m_id'],
                        'e1_trigger': m1['tokens'],
                        'e2_trigger': m2['tokens'],
                        'gold_label': gold,
                        'predicted_label': label,
                        'turns': turns,
                    }) + '\n')
                    tf.flush()

    predictions = {}
    for file in sorted(file_results.keys()):
        edges = []
        for idx in sorted(file_results[file].keys()):
            m1, m2, label = file_results[file][idx]
            edges.append((m1['tokens'], m1['m_id'], m2['tokens'], m2['m_id'], label))
        predictions[file] = {"target": build_dot(edges)}
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="together | gpt")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--q1_traces_file", required=True,
                        help="JSONL emitted by run_cot_yuan_q1only.py")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--matres_folder", default=DEFAULT_TRAIN_FOLDER)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--debug_first_messages", action='store_true',
                        help="Print self.messages before the first Q2 call (once, globally)")
    parser.add_argument("--smoke_balance", type=int, default=None,
                        help="Smoke-test mode: pick N/2 Q1=yes + N/2 Q1=no pairs and exit")
    args = parser.parse_args()

    if args.api == "gemini":
        raise SystemExit("Gemini not supported here; use a Gemini-specific runner.")
    if args.api not in ("together", "gpt"):
        raise SystemExit(f"Unknown --api: {args.api!r}")
    if not os.path.isdir(args.matres_folder):
        raise SystemExit(f"MATRES folder not found: {args.matres_folder}")
    if not os.path.isfile(args.q1_traces_file):
        raise SystemExit(f"Q1 traces file not found: {args.q1_traces_file}")

    if os.path.exists(args.output_folder):
        print("Output folder already exists. Exiting to avoid overwrite.", file=sys.stderr)
        sys.exit(0)
    os.makedirs(args.output_folder)

    short = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    tag = f"matres_train_continue_{short}"
    output_file = f"{args.output_folder}/{tag}.json"
    traces_file = f"{args.output_folder}/{tag}.traces.jsonl"

    start = time.time()
    main(api=args.api, model_name=args.model_name,
         q1_file=args.q1_traces_file, output_file=output_file,
         traces_file=traces_file, matres_folder=args.matres_folder,
         max_pairs=args.max_pairs, workers=args.workers,
         debug_first=args.debug_first_messages,
         smoke_balance=args.smoke_balance)
    print(f"Took {time.time() - start:.1f}s", file=sys.stderr)
