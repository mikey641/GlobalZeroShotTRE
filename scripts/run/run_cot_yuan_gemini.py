"""Yuan CoT runner for MATRES, using Gemini 2.5 Pro via Vertex AI + ADC.

Mirrors run_cot_yuan.py (same prompts, same Q1/branch-A/branch-B structure,
same sort order, same DOT output), but talks to Gemini through google-genai
chat sessions and records per-turn {question, thought, response, parsed}.

Auth: Application Default Credentials. `gcloud auth application-default login`.
Env: GOOGLE_CLOUD_PROJECT (required), GOOGLE_CLOUD_LOCATION (default us-central1).

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_cot_yuan_gemini.py \\
        --output_folder output/matres_gemini25pro_pilot50 \\
        --max_pairs 50 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tqdm import tqdm


RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
MAX_RETRIES = 6


def _send_with_retry(chat, question):
    """Retry transient Vertex errors (429 + 5xx) with capped exponential backoff."""
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return chat.send_message(question)
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            last = e
        except Exception as e:  # httpx ReadTimeout, ConnectionError, etc.
            if attempt == MAX_RETRIES:
                raise
            last = e
        # Exp backoff: 2, 4, 8, 16, 32, 64s, + 0-1s jitter.
        sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
        print(f"[retry {attempt+1}/{MAX_RETRIES}] {type(last).__name__}: sleeping {sleep_s:.1f}s",
              file=sys.stderr, flush=True)
        time.sleep(sleep_s)

from scripts.run.prompts_cot_yuan import (
    mark_target_pair_in_doc, ref,
    q_same_event,
    q_simultaneous_same_event, q_simultaneous,
    q_before_same_event, q_after_same_event,
    q_before, q_after,
    parse_yes_no,
)
from scripts.utils.io_utils import open_input_file


MATRES_TEST_FOLDER = 'data/MATRES/_in_OmniTemp_format/test'

_thread_local = threading.local()


def _thread_client(model_name):
    if not hasattr(_thread_local, 'client'):
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        _thread_local.client = genai.Client(vertexai=True, project=project, location=location)
        _thread_local.model_name = model_name
    return _thread_local.client


def _new_chat(client, model_name):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=-1,
            ),
        ),
    )


def _extract(response):
    """Return (thought_text, answer_text) from a Gemini response."""
    thoughts, answer = [], []
    cands = getattr(response, "candidates", None) or []
    if cands:
        parts = getattr(cands[0].content, "parts", None) or []
        for p in parts:
            t = getattr(p, "text", "") or ""
            if getattr(p, "thought", False):
                thoughts.append(t)
            else:
                answer.append(t)
    if not answer and not thoughts:
        answer.append(getattr(response, "text", "") or "")
    return "".join(thoughts), "".join(answer)


def _ask(chat, question, trace):
    resp = _send_with_retry(chat, question)
    thought, answer = _extract(resp)
    parsed = parse_yes_no(answer)
    trace.append({
        "question": question,
        "thought": thought,
        "response": answer,
        "parsed": parsed,
    })
    return parsed


def run_cot_pair(chat, doc_tokens, m1, m2):
    """Yuan CoT protocol for one pair. Returns (label, trace)."""
    doc_text = mark_target_pair_in_doc(
        doc_tokens, m1['tokens_ids'], m2['tokens_ids'],
        m1['m_id'], m2['m_id'],
    )
    e1 = ref(m1['tokens'], m1['m_id'])
    e2 = ref(m2['tokens'], m2['m_id'])

    trace = []

    same = _ask(chat, q_same_event(doc_text, e1, e2), trace)

    if same == 'yes':
        if _ask(chat, q_simultaneous_same_event(e1, e2), trace) == 'yes':
            return 'EQUAL', trace
        if _ask(chat, q_before_same_event(e1, e2), trace) == 'yes':
            return 'BEFORE', trace
        if _ask(chat, q_after_same_event(e1, e2), trace) == 'yes':
            return 'AFTER', trace
        return 'VAGUE', trace

    if _ask(chat, q_simultaneous(e1, e2), trace) == 'yes':
        return 'EQUAL', trace
    if _ask(chat, q_before(e1, e2), trace) == 'yes':
        return 'BEFORE', trace
    if _ask(chat, q_after(e1, e2), trace) == 'yes':
        return 'AFTER', trace
    return 'VAGUE', trace


def build_dot(edges):
    lines = ["strict graph {"]
    for m1_name, m1_id, m2_name, m2_id, label in edges:
        lines.append(f'"{m1_name}({m1_id})" -- "{m2_name}({m2_id})" [rel={label.lower()}];')
    lines.append("}")
    return "\n".join(lines)


def _sort_pairs_by_doc_position(all_pairs, ment_dict):
    return sorted(
        all_pairs,
        key=lambda p: ment_dict[p['_firstId']]['tokens_ids'][0],
    )


def _worker(task, model_name):
    file, pair_idx, doc_tokens, m1, m2, gold = task
    client = _thread_client(model_name)
    chat = _new_chat(client, model_name)
    label, trace = run_cot_pair(chat, doc_tokens, m1, m2)
    return file, pair_idx, m1, m2, gold, label, trace


def build_tasks(max_pairs):
    files = sorted(f for f in os.listdir(MATRES_TEST_FOLDER)
                   if f.endswith('.json') or f.endswith('.jsonl'))
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
    return tasks


def main(model_name, output_file, traces_file, max_pairs, workers):
    tasks = build_tasks(max_pairs)
    print(f"Loaded {len(tasks)} pairs from {MATRES_TEST_FOLDER}", file=sys.stderr)

    file_results = {}
    lock = threading.Lock()

    with open(traces_file, 'w') as tf:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, t, model_name) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stderr):
                try:
                    file, pair_idx, m1, m2, gold, label, trace = fut.result()
                except Exception as e:
                    print(f"Task failed: {e!r}", file=sys.stderr)
                    continue
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
                        'turns': trace,
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

    print(f"Wrote {len(predictions)} docs to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gemini-2.5-pro")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        raise SystemExit("GOOGLE_CLOUD_PROJECT env var is required (Vertex AI + ADC).")

    if os.path.exists(args.output_folder):
        print("Output folder already exists. Exiting to avoid overwrite.", file=sys.stderr)
        sys.exit(0)
    os.makedirs(args.output_folder)

    short = args.model_name
    tag = f"matres_cot_yuan_{short}"
    output_file = f"{args.output_folder}/{tag}.json"
    traces_file = f"{args.output_folder}/{tag}.traces.jsonl"

    start = time.time()
    main(
        model_name=args.model_name,
        output_file=output_file,
        traces_file=traces_file,
        max_pairs=args.max_pairs,
        workers=args.workers,
    )
    print(f"Took {time.time() - start:.1f}s", file=sys.stderr)
