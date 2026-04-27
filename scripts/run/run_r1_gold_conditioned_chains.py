"""STaR-style gold-conditioned chain generation for R1-failed pairs.

For each pair where DeepSeek-R1's natural Yuan-tree chain didn't reach gold:
  1. Reuse R1's correct prefix (turns 1..k-1 verbatim) up to the divergence point.
  2. Add a SYSTEM message instructing R1 to land on the gold label without naming it.
  3. From turn k onward, regenerate via Together API. Retry once with a hint on
     mismatch. Force the expected commit (preserving R1's <think>) if retry fails.
  4. Validate the merged chain lands on gold. Save in R1-trace-compatible JSONL
     PLUS extra metadata (divergence_turn, reused_prefix_length, forced_commits,
     leakage_flag, generation_mode, q1_source).

The system message is NEVER saved to the output file — only the conversation
turns are.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_r1_gold_conditioned_chains.py \\
        --output_file output/matres_train_continue_r1_goldconditioned_pilot50.jsonl \\
        --max_pairs 50 --workers 4 --stratified
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    q_simultaneous, q_simultaneous_same_event,
    q_before, q_before_same_event,
    q_after, q_after_same_event,
    parse_yes_no,
)
from scripts.utils.llms_definitions import TogetherModel

# ── retry on transient Together errors (mirrors run_cot_yuan_continue.py) ──
_RETRY_SUBSTRINGS = (
    'RateLimitError', 'ReadTimeout', 'InternalServerError',
    'RemoteProtocolError', 'APIConnectionError', 'ServiceUnavailableError',
    'APITimeoutError', 'ConnectError',
)
MAX_NETWORK_RETRIES = 6


def _is_retryable(err):
    n = type(err).__name__
    if any(s in n for s in _RETRY_SUBSTRINGS):
        return True
    sc = getattr(err, 'status_code', None)
    if sc in (408, 429, 500, 502, 503, 504):
        return True
    return False


def _chat_with_retry(llm, question):
    for attempt in range(MAX_NETWORK_RETRIES + 1):
        msgs_before = len(llm.messages)
        try:
            return llm.run_model_chat(question)
        except Exception as e:
            if len(llm.messages) > msgs_before:
                del llm.messages[msgs_before:]
            if not _is_retryable(e) or attempt == MAX_NETWORK_RETRIES:
                raise
            sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
            print(f'[retry {attempt+1}] {type(e).__name__}: sleep {sleep_s:.1f}s',
                  file=sys.stderr, flush=True)
            time.sleep(sleep_s)


# ── R1 trace + gold-walk logic ──
R1_TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'

# Maps gold label → list of (q_id, expected_parsed) for Q2/Q3/Q4
GOLD_WALK = {
    'EQUAL':  [('Q2', 'yes')],
    'BEFORE': [('Q2', 'no'), ('Q3', 'yes')],
    'AFTER':  [('Q2', 'no'), ('Q3', 'no'), ('Q4', 'yes')],
    'VAGUE':  [('Q2', 'no'), ('Q3', 'no'), ('Q4', 'no')],
}


def question_for(q_id, e1_ref, e2_ref, q1_yes):
    if q_id == 'Q2':
        return q_simultaneous_same_event(e1_ref, e2_ref) if q1_yes else q_simultaneous(e1_ref, e2_ref)
    if q_id == 'Q3':
        return q_before_same_event(e1_ref, e2_ref) if q1_yes else q_before(e1_ref, e2_ref)
    if q_id == 'Q4':
        return q_after_same_event(e1_ref, e2_ref) if q1_yes else q_after(e1_ref, e2_ref)
    raise ValueError(q_id)


def find_divergence(r1_turns, gold_label):
    """Return divergence_turn_idx (0-indexed: Q2=1, Q3=2, Q4=3), or None.

    'None' means R1's chain matches gold-walk fully — should not happen for
    R1-failed pairs (data integrity violation).
    """
    expected = GOLD_WALK[gold_label]
    for i, (q_id, exp) in enumerate(expected):
        turn_idx = 1 + i
        if turn_idx >= len(r1_turns):
            return turn_idx  # R1 stopped early — divergence here
        actual = r1_turns[turn_idx].get('parsed')
        if actual != exp:
            return turn_idx
    return None


# ── leakage scan on regenerated <think> blocks ──
LEAKAGE_PATTERNS = [
    re.compile(r'\bBEFORE\b', re.IGNORECASE),
    re.compile(r'\bAFTER\b', re.IGNORECASE),
    re.compile(r'\bEQUAL\b', re.IGNORECASE),
    re.compile(r'\bVAGUE\b', re.IGNORECASE),
    re.compile(r'\bgold\s+label\b', re.IGNORECASE),
    re.compile(r'\bthe\s+label\s+is\b', re.IGNORECASE),
    re.compile(r'\bthe\s+answer\s+is\s+(BEFORE|AFTER|EQUAL|VAGUE)\b', re.IGNORECASE),
]


def has_leakage(think_text):
    if not think_text:
        return False
    return any(p.search(think_text) for p in LEAKAGE_PATTERNS)


# ── extract <think>...</think> from R1 response ──
THINK_RE = re.compile(r'<think>(.*?)</think>\s*', re.DOTALL | re.IGNORECASE)


def extract_think(response_text):
    """Return (think_content, answer_text). Uses the LAST </think> as boundary."""
    if not response_text:
        return '', ''
    m = THINK_RE.search(response_text)
    if not m:
        return '', response_text.strip()
    think = m.group(1).strip()
    answer = response_text[m.end():].strip()
    return think, answer


# ── system message (NOT saved with output) ──
def build_system_message(e1_trigger, e2_trigger, gold_label):
    return (
        f"For this temporal relation extraction problem, the gold label between "
        f"{e1_trigger} and {e2_trigger} is {gold_label}. Generate Yuan-tree reasoning "
        f"that walks naturally to this label. Reason through each question on its own "
        f"merits without explicitly stating the label or its name in your reasoning. "
        f"Your yes/no commits should land on the path that produces this label."
    )


# ── core: regenerate divergent suffix for one pair ──
def generate_one_pair(llm, r1_row, divergence_idx, q1_yes):
    """Generate gold-conditioned suffix from divergence_idx onward.

    Returns dict with merged chain + metadata.
    """
    gold = r1_row['gold_label']
    expected_walk = GOLD_WALK[gold]  # [(Q2, exp), (Q3, exp), (Q4, exp)] truncated by gold

    # Reused prefix: R1 turns [0..divergence_idx-1]
    r1_turns = r1_row['turns']
    reused = r1_turns[:divergence_idx]

    # Build e_ref strings (from triggers in the row)
    e1_ref = f"<EVENT e{r1_row['e1_id']}>{r1_row['e1_trigger']}</EVENT>"
    e2_ref = f"<EVENT e{r1_row['e2_id']}>{r1_row['e2_trigger']}</EVENT>"

    # Seed conversation: system message + reused prefix turns as user/assistant pairs
    llm.clear()
    llm.messages.append({
        'role': 'system',
        'content': build_system_message(r1_row['e1_trigger'], r1_row['e2_trigger'], gold),
    })
    for t in reused:
        llm.messages.append({'role': 'user', 'content': t['question']})
        llm.messages.append({'role': 'assistant', 'content': t['response']})

    # Generate turns from divergence_idx onward
    new_turns = []
    forced_commits = []
    leakage_flag = False
    natural_first_try_count = 0
    natural_total = 0

    # Determine how many new turns we need (from gold-walk, indices >= divergence_idx-1)
    # divergence_idx is 1-indexed turn position in the chain (Q2=1, Q3=2, Q4=3),
    # but expected_walk is 0-indexed in the list. Q2 commit is expected_walk[0].
    # So expected_walk index = divergence_idx - 1.
    walk_start = divergence_idx - 1
    if walk_start < 0 or walk_start >= len(expected_walk):
        return {'error': f'invalid walk_start {walk_start}, expected_walk len {len(expected_walk)}'}

    for j in range(walk_start, len(expected_walk)):
        q_id, expected_parsed = expected_walk[j]
        question = question_for(q_id, e1_ref, e2_ref, q1_yes)

        # Attempt 1: natural
        natural_total += 1
        out = _chat_with_retry(llm, question)
        think, answer = extract_think(out)
        parsed = parse_yes_no(answer)
        forced = False

        if parsed == expected_parsed:
            natural_first_try_count += 1
        else:
            # Retry once with hint
            hint = (f"Hint: based on the document evidence, the answer to this "
                    f"question should be {'yes' if expected_parsed == 'yes' else 'no'}.")
            out2 = _chat_with_retry(llm, hint)
            think2, answer2 = extract_think(out2)
            parsed2 = parse_yes_no(answer2)

            if parsed2 == expected_parsed:
                # Retry succeeded — replace the original turn with the retry's content.
                # llm.messages now has [..., user:question, asst:wrong_out, user:hint, asst:out2].
                # We collapse it: pop the wrong & hint, keep new asst attached to original question.
                # Specifically: keep messages [..., user:question, asst:out2].
                # Pop the wrong assistant + hint user + retry assistant + restore retry assistant under original question.
                # llm.messages[-4] = user(question), [-3] = asst(wrong), [-2] = user(hint), [-1] = asst(retry).
                # Replace [-3] with the retry, drop [-2] and [-1].
                if len(llm.messages) >= 4:
                    llm.messages[-3] = {'role': 'assistant', 'content': out2}
                    del llm.messages[-2:]
                think, answer, parsed = think2, answer2, parsed2
                out = out2
            else:
                # Force the commit: keep retry's <think>, override answer text.
                # llm.messages currently has [..., user:question, asst:out, user:hint, asst:out2].
                # Reshape so the chat log shows [user:question, asst:forced_response].
                forced_answer = 'Yes.' if expected_parsed == 'yes' else 'No.'
                forced_response = f'<think>{think2}</think>\n\n{forced_answer}' if think2 else forced_answer
                if len(llm.messages) >= 4:
                    llm.messages[-3] = {'role': 'assistant', 'content': forced_response}
                    del llm.messages[-2:]
                think, answer, parsed = think2, forced_answer, expected_parsed
                out = forced_response
                forced = True
                forced_commits.append(j + 1)  # 1-indexed turn in walk

        # Leakage scan on the regenerated <think>
        if has_leakage(think):
            leakage_flag = True

        new_turns.append({
            'question': question,
            'think': think,
            'response': out,
            'parsed': parsed,
            'forced': forced,
        })

        # If this is a terminal Yes/No that ends the gold-walk, stop generating
        # (e.g. EQUAL chains stop after Q2=Yes).
        if expected_parsed == 'yes' and q_id in ('Q2', 'Q3', 'Q4'):
            break
        if q_id == 'Q4' and expected_parsed == 'no':
            break  # VAGUE terminal

    # Build the full merged chain for the saved trace
    merged_turns = []
    for t in reused:
        rthink, ranswer = extract_think(t.get('response', ''))
        merged_turns.append({
            'question': t['question'],
            'think': rthink,
            'response': t.get('response', ''),
            'parsed': t.get('parsed'),
        })
    merged_turns.extend(new_turns)

    return {
        'merged_turns': merged_turns,
        'reused_prefix_length': len(reused),
        'divergence_turn': divergence_idx,
        'forced_commits': forced_commits,
        'leakage_flag': leakage_flag,
        'natural_first_try_count': natural_first_try_count,
        'natural_total': natural_total,
        'error': None,
    }


# ── threadlocal LLM (one connection per worker) ──
_thread_local = threading.local()


def _thread_llm(model_name):
    if not hasattr(_thread_local, 'llm'):
        _thread_local.llm = TogetherModel(model_name)
    return _thread_local.llm


# ── load + filter R1 traces ──
def load_r1_failed():
    rows = []
    with open(R1_TRACES) as f:
        for line in f:
            t = json.loads(line)
            if t.get('predicted_label') != t.get('gold_label'):
                rows.append(t)
    return rows


def stratified_sample(rows, n, seed=0):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r['gold_label']].append(r)
    total = len(rows)
    quotas = {c: round(n * len(rs) / total) for c, rs in by_class.items()}
    diff = n - sum(quotas.values())
    if diff != 0:
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


def _worker(r1_row, model_name, suppress_stdout):
    llm = _thread_llm(model_name)
    # Determine Q1 branch from R1's saved Q1 turn
    if not r1_row.get('turns'):
        return r1_row, {'error': 'no turns in R1 row'}
    q1_parsed = r1_row['turns'][0].get('parsed')
    q1_yes = (q1_parsed == 'yes')

    gold = r1_row['gold_label']
    div_idx = find_divergence(r1_row['turns'], gold)

    if div_idx is None:
        return r1_row, {'error': 'no divergence (R1 chain matches gold-walk — unexpected for R1-failed pair)'}
    if div_idx == 0:
        return r1_row, {'error': 'divergence at Q1 — skipped per spec', 'q1_divergence': True}

    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            result = generate_one_pair(llm, r1_row, div_idx, q1_yes)
    else:
        result = generate_one_pair(llm, r1_row, div_idx, q1_yes)
    return r1_row, result


def main(model_name, output_file, max_pairs, workers, stratified):
    rows = load_r1_failed()
    print(f'R1-failed pairs total: {len(rows)}', file=sys.stderr, flush=True)

    if stratified and max_pairs:
        sample = stratified_sample(rows, max_pairs)
        cls_dist = Counter(r['gold_label'] for r in sample)
        print(f'Stratified sample of {len(sample)}: {dict(cls_dist)}', file=sys.stderr, flush=True)
    elif max_pairs:
        sample = rows[:max_pairs]
    else:
        sample = rows

    # Pre-compute divergence distribution over the sample (no API calls)
    pre_div = Counter()
    pre_q1 = Counter()
    pre_no_div = 0
    for r in sample:
        if not r.get('turns'):
            continue
        q1p = r['turns'][0].get('parsed')
        pre_q1[q1p] += 1
        d = find_divergence(r['turns'], r['gold_label'])
        if d is None:
            pre_no_div += 1
        elif d == 0:
            pre_div['Q1'] += 1
        elif d == 1:
            pre_div['Q2'] += 1
        elif d == 2:
            pre_div['Q3'] += 1
        elif d == 3:
            pre_div['Q4'] += 1
    print(f'Pre-API divergence distribution: {dict(pre_div)}', file=sys.stderr, flush=True)
    print(f'Pre-API Q1 parse distribution:   {dict(pre_q1)}', file=sys.stderr, flush=True)
    print(f'Pre-API no-divergence (data err): {pre_no_div}', file=sys.stderr, flush=True)

    # Filter out Q1-divergence + no-divergence pairs (skip per spec)
    runnable = []
    skipped_q1 = 0
    skipped_nodiv = 0
    for r in sample:
        if not r.get('turns'):
            continue
        d = find_divergence(r['turns'], r['gold_label'])
        if d is None:
            skipped_nodiv += 1
            continue
        if d == 0:
            skipped_q1 += 1
            continue
        runnable.append(r)
    print(f'Runnable: {len(runnable)} (skipped Q1-div: {skipped_q1}, no-div: {skipped_nodiv})',
          file=sys.stderr, flush=True)

    # Generate
    success = 0
    errored = 0
    forced_pair_count = 0
    leakage_count = 0
    natural_total_all = 0
    natural_correct_all = 0
    forced_turn_total = 0
    div_per_turn = Counter()
    reused_len_dist = Counter()

    suppress = workers > 1
    lock = threading.Lock()
    start = time.time()

    with open(output_file, 'w') as out_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_worker, r, model_name, suppress) for r in runnable]
            for fut in tqdm(as_completed(futs), total=len(futs), file=sys.stderr):
                try:
                    r1_row, result = fut.result()
                except Exception as e:
                    print(f'task failed: {e!r}', file=sys.stderr, flush=True)
                    errored += 1
                    continue
                if result.get('error'):
                    errored += 1
                    print(f'  err {r1_row["doc_id"]} e{r1_row["e1_id"]}-{r1_row["e2_id"]}: {result["error"][:100]}',
                          file=sys.stderr, flush=True)
                    continue

                # Validate merged chain lands on gold
                merged = result['merged_turns']
                derived = derive_label_from_chain(merged)
                if derived != r1_row['gold_label']:
                    print(f'  validate FAIL {r1_row["doc_id"]} e{r1_row["e1_id"]}-{r1_row["e2_id"]}: '
                          f'derived={derived} gold={r1_row["gold_label"]}', file=sys.stderr, flush=True)
                    errored += 1
                    continue

                success += 1
                if result['forced_commits']:
                    forced_pair_count += 1
                    forced_turn_total += len(result['forced_commits'])
                if result['leakage_flag']:
                    leakage_count += 1
                natural_total_all += result['natural_total']
                natural_correct_all += result['natural_first_try_count']
                div_per_turn[result['divergence_turn']] += 1
                reused_len_dist[result['reused_prefix_length']] += 1

                # Write the row
                row_out = {
                    'doc_id': r1_row['doc_id'],
                    'e1_id': r1_row['e1_id'],
                    'e2_id': r1_row['e2_id'],
                    'e1_trigger': r1_row['e1_trigger'],
                    'e2_trigger': r1_row['e2_trigger'],
                    'gold_label': r1_row['gold_label'],
                    'predicted_label': r1_row['gold_label'],  # by construction
                    'turns': merged,
                    'generation_mode': 'gold_conditioned_with_prefix_reuse',
                    'divergence_turn': result['divergence_turn'],
                    'reused_prefix_length': result['reused_prefix_length'],
                    'forced_commits': result['forced_commits'],
                    'leakage_flag': result['leakage_flag'],
                    'q1_source': 'r1_forward_pass',
                }
                with lock:
                    out_f.write(json.dumps(row_out) + '\n')
                    out_f.flush()

    wall = time.time() - start
    print('\n========= report =========', file=sys.stderr, flush=True)
    print(f'wall:                  {wall/60:.1f} min', file=sys.stderr, flush=True)
    print(f'attempted:             {len(runnable)}', file=sys.stderr, flush=True)
    print(f'success:               {success}', file=sys.stderr, flush=True)
    print(f'errored/skipped:       {errored}', file=sys.stderr, flush=True)
    print(f'skipped (Q1-div):      {skipped_q1}', file=sys.stderr, flush=True)
    print(f'skipped (no-div):      {skipped_nodiv}', file=sys.stderr, flush=True)
    print(f'div distribution:     {dict(div_per_turn)} (1=Q2, 2=Q3, 3=Q4)', file=sys.stderr, flush=True)
    print(f'reused prefix length: {dict(reused_len_dist)}', file=sys.stderr, flush=True)
    if natural_total_all:
        print(f'natural-agreement:    {natural_correct_all}/{natural_total_all} = '
              f'{100*natural_correct_all/natural_total_all:.1f}% (per-turn first-try match)',
              file=sys.stderr, flush=True)
    print(f'pairs with any forced commit: {forced_pair_count}/{success} = '
          f'{100*forced_pair_count/max(1,success):.1f}%', file=sys.stderr, flush=True)
    print(f'total forced turns:   {forced_turn_total}', file=sys.stderr, flush=True)
    print(f'leakage_flag pairs:   {leakage_count}/{success} = '
          f'{100*leakage_count/max(1,success):.1f}%', file=sys.stderr, flush=True)


def derive_label_from_chain(turns):
    """Walk turns[1:] (skip Q1) and return the label per Yuan tree."""
    if len(turns) < 2:
        return None
    # Q2 commit
    q2 = turns[1].get('parsed')
    if q2 == 'yes':
        return 'EQUAL'
    if q2 != 'no':
        return None
    if len(turns) < 3:
        return None
    q3 = turns[2].get('parsed')
    if q3 == 'yes':
        return 'BEFORE'
    if q3 != 'no':
        return None
    if len(turns) < 4:
        return None
    q4 = turns[3].get('parsed')
    if q4 == 'yes':
        return 'AFTER'
    if q4 == 'no':
        return 'VAGUE'
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stratified', action='store_true')
    args = parser.parse_args()

    if not os.environ.get('TOGETHER_API_KEY'):
        raise SystemExit('TOGETHER_API_KEY env var required')
    if os.path.exists(args.output_file):
        raise SystemExit(f'Output exists: {args.output_file}')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args.model_name, args.output_file, args.max_pairs, args.workers, args.stratified)
