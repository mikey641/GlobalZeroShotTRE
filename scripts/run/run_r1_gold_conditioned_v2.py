"""v2: STaR-style gold-conditioned chain generation via per-turn hints (no system msg).

Differences from v1 (run_r1_gold_conditioned_chains.py):
  - NO system message anywhere.
  - Hints are appended inline to user-turn questions: "{question} Hint: the
    answer is {yes|no}."
  - At divergence turn k: send WITH hint (we know R1 was already wrong here).
    On mismatch retry once with stronger hint, then force.
  - At turn k+1, k+2 (if gold-walk requires): try NATURAL first. Only add
    hint if R1 commits wrong; same retry+force ladder as turn k.
  - Hints are STRIPPED from the saved user-message text. Saved JSONL has the
    bare question, plus metadata: hint_applied (list of turn idxs that needed
    a hint during generation), forced_commits (list of turn idxs forced).
  - Type-B-only leakage detector: catches "gold label", capitalized label
    naming ("the answer is BEFORE", "BEFORE label", etc.), explicit "hint"
    mentions. Does NOT flag casual lowercase before/after.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. python scripts/run/run_r1_gold_conditioned_v2.py \\
        --output_file output/matres_train_continue_r1_goldconditioned_v2_pilot50.jsonl \\
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
from typing import List, Tuple

from tqdm import tqdm

from scripts.run.prompts_cot_yuan import (
    q_simultaneous, q_simultaneous_same_event,
    q_before, q_before_same_event,
    q_after, q_after_same_event,
    parse_yes_no,
)
from scripts.utils.llms_definitions import TogetherModel


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


def _chat_with_retry(llm, message_text):
    for attempt in range(MAX_NETWORK_RETRIES + 1):
        msgs_before = len(llm.messages)
        try:
            return llm.run_model_chat(message_text)
        except Exception as e:
            if len(llm.messages) > msgs_before:
                del llm.messages[msgs_before:]
            if not _is_retryable(e) or attempt == MAX_NETWORK_RETRIES:
                raise
            sleep_s = min(64, 2 ** (attempt + 1)) + random.random()
            print(f'[net-retry {attempt+1}] {type(e).__name__}: sleep {sleep_s:.1f}s',
                  file=sys.stderr, flush=True)
            time.sleep(sleep_s)


R1_TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'

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
    expected = GOLD_WALK[gold_label]
    for i, (q_id, exp) in enumerate(expected):
        turn_idx = 1 + i
        if turn_idx >= len(r1_turns):
            return turn_idx
        actual = r1_turns[turn_idx].get('parsed')
        if actual != exp:
            return turn_idx
    return None


# Type-B-only leakage detector — catches *label-naming* and "hint" mentions.
# Does NOT flag casual lowercase "before"/"after" (those are valid English in
# temporal reasoning).
LEAKAGE_PATTERNS = [
    re.compile(r'\b(the\s+)?gold\s+label\b', re.IGNORECASE),
    re.compile(r'\bthe\s+answer\s+is\s+(BEFORE|AFTER|EQUAL|VAGUE)\b'),  # caps required
    re.compile(r'\bthe\s+(temporal\s+)?(relation|label)\s+is\s+(BEFORE|AFTER|EQUAL|VAGUE)\b'),
    re.compile(r'\b(this|it)\s+aligns?\s+with\s+(the\s+)?(BEFORE|AFTER|EQUAL|VAGUE)\b'),
    re.compile(r'\b(BEFORE|AFTER|EQUAL|VAGUE)\s+label\b'),
    re.compile(r'\blabel\s+(BEFORE|AFTER|EQUAL|VAGUE)\b'),
    re.compile(r'\bhint\b', re.IGNORECASE),  # any "hint" mention is leakage from our prompt
]


def has_leakage(think_text):
    if not think_text:
        return False
    return any(p.search(think_text) for p in LEAKAGE_PATTERNS)


THINK_RE = re.compile(r'<think>(.*?)</think>\s*', re.DOTALL | re.IGNORECASE)


def extract_think(response_text):
    if not response_text:
        return '', ''
    m = THINK_RE.search(response_text)
    if not m:
        return '', response_text.strip()
    return m.group(1).strip(), response_text[m.end():].strip()


def hint_msg(question, expected_parsed, kind):
    yn = 'yes' if expected_parsed == 'yes' else 'no'
    if kind == 'weak':
        return f"{question} Hint: the answer is {yn}."
    if kind == 'strong':
        return f"{question} Hint: based on document evidence, the answer is {yn}."
    raise ValueError(kind)


def _rollback_last_pair(llm):
    """Remove the most recent (user, assistant) message pair."""
    if len(llm.messages) >= 2:
        del llm.messages[-2:]


def make_attempts(llm, question, expected_parsed, attempts: List[Tuple[str, str]]):
    """Try each attempt in order. On match, save with BARE question. On failure
    of all attempts, force the commit.

    attempts: list of ('natural'|'weak'|'strong', message_text_to_send).
    Returns: dict with think, response, parsed, hint_applied, forced.
    """
    last_think = ''
    last_response = ''
    for kind, msg_text in attempts:
        out = _chat_with_retry(llm, msg_text)
        think, answer = extract_think(out)
        parsed = parse_yes_no(answer)
        last_think = think
        last_response = out
        if parsed == expected_parsed:
            # Roll back the hinted message + replace with bare question
            _rollback_last_pair(llm)
            llm.messages.append({'role': 'user', 'content': question})
            llm.messages.append({'role': 'assistant', 'content': out})
            return {
                'think': think, 'response': out, 'parsed': parsed,
                'hint_applied': kind != 'natural', 'forced': False,
            }
        # Mismatch — roll back for next attempt
        _rollback_last_pair(llm)

    # All attempts exhausted — force using last attempt's <think>
    forced_answer_text = 'Yes.' if expected_parsed == 'yes' else 'No.'
    forced_response = (f'<think>{last_think}</think>\n\n{forced_answer_text}'
                       if last_think else forced_answer_text)
    llm.messages.append({'role': 'user', 'content': question})
    llm.messages.append({'role': 'assistant', 'content': forced_response})
    return {
        'think': last_think, 'response': forced_response, 'parsed': expected_parsed,
        'hint_applied': True, 'forced': True,
    }


def generate_one_pair(llm, r1_row, divergence_idx, q1_yes):
    gold = r1_row['gold_label']
    expected_walk = GOLD_WALK[gold]

    # Reused prefix: R1 turns [0..divergence_idx-1]
    r1_turns = r1_row['turns']
    reused = r1_turns[:divergence_idx]

    e1_ref = f"<EVENT e{r1_row['e1_id']}>{r1_row['e1_trigger']}</EVENT>"
    e2_ref = f"<EVENT e{r1_row['e2_id']}>{r1_row['e2_trigger']}</EVENT>"

    # Seed conversation: NO system message; just reused prefix
    llm.clear()
    for t in reused:
        llm.messages.append({'role': 'user', 'content': t['question']})
        llm.messages.append({'role': 'assistant', 'content': t['response']})

    new_turns = []
    forced_commits = []
    hint_applied_turns = []
    natural_first_try_count = 0
    natural_total = 0  # turns where we tried natural at all

    walk_start = divergence_idx - 1
    if walk_start < 0 or walk_start >= len(expected_walk):
        return {'error': f'invalid walk_start {walk_start}'}

    for j in range(walk_start, len(expected_walk)):
        q_id, expected_parsed = expected_walk[j]
        question = question_for(q_id, e1_ref, e2_ref, q1_yes)
        is_divergence_turn = (j == walk_start)

        # Build attempt ladder
        if is_divergence_turn:
            # We KNOW R1 was wrong here in forward pass — skip natural, start with hint.
            attempts = [
                ('weak',   hint_msg(question, expected_parsed, 'weak')),
                ('strong', hint_msg(question, expected_parsed, 'strong')),
            ]
        else:
            # Try natural first; hint only if R1 commits wrong.
            attempts = [
                ('natural', question),
                ('weak',    hint_msg(question, expected_parsed, 'weak')),
                ('strong',  hint_msg(question, expected_parsed, 'strong')),
            ]
            natural_total += 1

        result = make_attempts(llm, question, expected_parsed, attempts)
        turn_idx_1based = j + 1  # 1-indexed turn position in walk (Q2=1, Q3=2, Q4=3)
        if not is_divergence_turn and not result['hint_applied']:
            natural_first_try_count += 1
        if result['hint_applied']:
            hint_applied_turns.append(turn_idx_1based)
        if result['forced']:
            forced_commits.append(turn_idx_1based)

        new_turns.append({
            'question': question,
            'think': result['think'],
            'response': result['response'],
            'parsed': result['parsed'],
            'forced': result['forced'],
            'hint_applied': result['hint_applied'],
        })

        # Stop on terminal commits (Q2=Yes/Q3=Yes/Q4=Yes/Q4=No are all terminals already
        # because they come from gold-walk; the loop ends naturally).

    # Build merged chain
    merged_turns = []
    for t in reused:
        rthink, _ = extract_think(t.get('response', ''))
        merged_turns.append({
            'question': t['question'],
            'think': rthink,
            'response': t.get('response', ''),
            'parsed': t.get('parsed'),
        })
    merged_turns.extend(new_turns)

    # Leakage scan over GENERATED <think> blocks only (not reused R1 ones)
    leakage_flag = any(has_leakage(t['think']) for t in new_turns)

    return {
        'merged_turns': merged_turns,
        'reused_prefix_length': len(reused),
        'divergence_turn': divergence_idx,
        'forced_commits': forced_commits,
        'hint_applied_turns': hint_applied_turns,
        'leakage_flag': leakage_flag,
        'natural_first_try_count': natural_first_try_count,
        'natural_total': natural_total,
        'error': None,
    }


_thread_local = threading.local()


def _thread_llm(model_name):
    if not hasattr(_thread_local, 'llm'):
        _thread_local.llm = TogetherModel(model_name)
    return _thread_local.llm


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
    if not r1_row.get('turns'):
        return r1_row, {'error': 'no turns in R1 row'}
    q1_yes = (r1_row['turns'][0].get('parsed') == 'yes')
    div_idx = find_divergence(r1_row['turns'], r1_row['gold_label'])
    if div_idx is None:
        return r1_row, {'error': 'no divergence (unexpected for R1-failed pair)'}
    if div_idx == 0:
        return r1_row, {'error': 'Q1 divergence — skipped per spec'}

    if suppress_stdout:
        with contextlib.redirect_stdout(io.StringIO()):
            return r1_row, generate_one_pair(llm, r1_row, div_idx, q1_yes)
    return r1_row, generate_one_pair(llm, r1_row, div_idx, q1_yes)


def derive_label_from_chain(turns):
    if len(turns) < 2:
        return None
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


def main(model_name, output_file, max_pairs, workers, stratified):
    rows = load_r1_failed()
    print(f'R1-failed pairs total: {len(rows)}', file=sys.stderr, flush=True)

    if stratified and max_pairs:
        sample = stratified_sample(rows, max_pairs)
        print(f'Stratified sample of {len(sample)}: {dict(Counter(r["gold_label"] for r in sample))}',
              file=sys.stderr, flush=True)
    elif max_pairs:
        sample = rows[:max_pairs]
    else:
        sample = rows

    runnable = [r for r in sample
                if r.get('turns') and find_divergence(r['turns'], r['gold_label']) not in (None, 0)]
    skipped = len(sample) - len(runnable)

    success = 0
    errored = 0
    forced_pair_count = 0
    leakage_count = 0
    natural_total_all = 0
    natural_correct_all = 0
    forced_turn_total = 0
    hint_per_turn_idx = Counter()  # turn 1-indexed -> count of pairs with hint at that turn
    natural_per_turn_idx = Counter()  # turn 1-indexed -> count where natural worked
    natural_attempted_per_turn_idx = Counter()
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
                    print(f'  err {r1_row["doc_id"]} e{r1_row["e1_id"]}-{r1_row["e2_id"]}: {result["error"][:80]}',
                          file=sys.stderr, flush=True)
                    continue

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
                for ht in result['hint_applied_turns']:
                    hint_per_turn_idx[ht] += 1

                # natural attempts: each non-divergence turn we attempt natural first
                # walk_start = divergence_idx - 1; non-div turns are j > walk_start
                # turn idx in walk (1-indexed, Q2=1) for j=walk_start+1 to len(expected_walk)-1
                gold = r1_row['gold_label']
                walk_len = len(GOLD_WALK[gold])
                walk_start = result['divergence_turn'] - 1
                for j in range(walk_start + 1, walk_len):
                    natural_attempted_per_turn_idx[j + 1] += 1
                    # natural succeeded iff this turn idx is NOT in hint_applied_turns
                    if (j + 1) not in result['hint_applied_turns']:
                        natural_per_turn_idx[j + 1] += 1

                row_out = {
                    'doc_id': r1_row['doc_id'],
                    'e1_id': r1_row['e1_id'],
                    'e2_id': r1_row['e2_id'],
                    'e1_trigger': r1_row['e1_trigger'],
                    'e2_trigger': r1_row['e2_trigger'],
                    'gold_label': r1_row['gold_label'],
                    'predicted_label': r1_row['gold_label'],
                    'turns': merged,
                    'generation_mode': 'gold_conditioned_v2_per_turn_hint',
                    'divergence_turn': result['divergence_turn'],
                    'reused_prefix_length': result['reused_prefix_length'],
                    'forced_commits': result['forced_commits'],
                    'hint_applied_turns': result['hint_applied_turns'],
                    'leakage_flag': result['leakage_flag'],
                    'q1_source': 'r1_forward_pass',
                }
                with lock:
                    out_f.write(json.dumps(row_out) + '\n')
                    out_f.flush()

    wall = time.time() - start
    print('\n========= report =========', file=sys.stderr, flush=True)
    print(f'wall:                  {wall/60:.1f} min', file=sys.stderr, flush=True)
    print(f'attempted runnable:    {len(runnable)}  (skipped: {skipped})', file=sys.stderr, flush=True)
    print(f'success:               {success}', file=sys.stderr, flush=True)
    print(f'errored:               {errored}', file=sys.stderr, flush=True)
    print(f'div distribution:      {dict(div_per_turn)} (1=Q2, 2=Q3, 3=Q4)', file=sys.stderr, flush=True)
    print(f'reused prefix length:  {dict(reused_len_dist)}', file=sys.stderr, flush=True)
    print(f'pairs with any forced commit: {forced_pair_count}/{success} = '
          f'{100*forced_pair_count/max(1,success):.1f}%', file=sys.stderr, flush=True)
    print(f'total forced turns:    {forced_turn_total}', file=sys.stderr, flush=True)
    print(f'leakage_flag pairs:    {leakage_count}/{success} = '
          f'{100*leakage_count/max(1,success):.1f}%', file=sys.stderr, flush=True)
    print(f'hint-applied per-turn (turn 1=Q2 commit, 2=Q3 commit, 3=Q4 commit):',
          file=sys.stderr, flush=True)
    for t_idx in sorted(hint_per_turn_idx):
        print(f'  Q{t_idx + 1}: {hint_per_turn_idx[t_idx]}/{success} = '
              f'{100*hint_per_turn_idx[t_idx]/max(1,success):.1f}%',
              file=sys.stderr, flush=True)
    print(f'per-turn natural-agreement (excluding divergence turn):',
          file=sys.stderr, flush=True)
    for t_idx in sorted(natural_attempted_per_turn_idx):
        denom = natural_attempted_per_turn_idx[t_idx]
        ok = natural_per_turn_idx.get(t_idx, 0)
        print(f'  Q{t_idx + 1}: {ok}/{denom} = {100*ok/max(1,denom):.1f}%',
              file=sys.stderr, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-ai/DeepSeek-R1')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stratified', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TOGETHER_API_KEY'):
        raise SystemExit('TOGETHER_API_KEY required')
    if os.path.exists(args.output_file):
        raise SystemExit(f'Output exists: {args.output_file}')
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args.model_name, args.output_file, args.max_pairs, args.workers, args.stratified)
