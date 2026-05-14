"""Build v4 multi-turn training data files (v4a/v4b/v4c/v4d).

Source traces: output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl
Source docs:   data/MATRES/_in_OmniTemp_format/train/

Outputs:
  output/matres_train_v4a_full_directlabels.jsonl     (full data, no-think, teacher+gold-derived)
  output/matres_train_v4b_subset_noThink.jsonl        (teacher-correct subset, no-think)
  output/matres_train_v4c_subset_withThink.jsonl      (teacher-correct subset, <think>+label)

v4d (separate flag --v4d):
  Source: output/matres_train_v4d_full_reasoning_goldcond.jsonl  (raw merged, 11,885 rows)
  Output: output/matres_train_v4d_sft_format.jsonl
  Same chat-format as v4c (<think>...</Think>\\n\\nYes|No assistant), reasoning everywhere.

Does NOT upload. Does NOT launch fine-tunes.
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
import subprocess
import sys
from collections import Counter

from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc, ref


TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'
DOC_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'

OUT_V4A = 'output/matres_train_v4a_full_directlabels.jsonl'
OUT_V4B = 'output/matres_train_v4b_subset_noThink.jsonl'
OUT_V4C = 'output/matres_train_v4c_subset_withThink.jsonl'

THINK_RE = re.compile(r'<think>(.*?)</think>', re.IGNORECASE | re.DOTALL)
KEEP_SHORT_RE = re.compile(r'\s*Keep the answer short and concise\.\s*$')


def strip_keep_short(s):
    return KEEP_SHORT_RE.sub('', s).rstrip()


def load_doc_index():
    docs = {}
    for fn in sorted(os.listdir(DOC_FOLDER)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(DOC_FOLDER, fn)) as f:
            d = json.load(f)
        ment_by_id = {m['m_id']: m for m in d.get('allMentions', [])}
        docs[fn] = {'tokens': d['tokens'], 'mentions': ment_by_id}
    return docs


def construct_q1(marked_doc, e1_ref, e2_ref):
    return (f"Given the following document:\n\n{marked_doc}\n\n"
            f"Are {e1_ref} and {e2_ref} referring to the same event?")


def construct_q2(e1_ref, e2_ref, in_event):
    suffix = ' in that event' if in_event else ''
    return f"Did {e1_ref} and {e2_ref} simultaneously happen{suffix}?"


def construct_q3(e1_ref, e2_ref, in_event):
    suffix = ' in that event' if in_event else ''
    return f"Is {e1_ref} before {e2_ref}{suffix}?"


def construct_q4(e1_ref, e2_ref, in_event):
    suffix = ' in that event' if in_event else ''
    return f"Is {e1_ref} after {e2_ref}{suffix}?"


def yes_no_str(parsed):
    return 'Yes' if parsed == 'yes' else 'No'


def chain_for_gold(gold, q1_answer='no'):
    """Return list of (qid, parsed) for gold-derived chain.

    Q1 answer is supplied (typically from teacher; MATRES has no gold for same-event).
    Q2/Q3/Q4 follow the inverse Yuan tree to land on `gold`.
    """
    if q1_answer not in ('yes', 'no'):
        return None
    if gold == 'EQUAL':
        return [('Q1', q1_answer), ('Q2', 'yes')]
    if gold == 'BEFORE':
        return [('Q1', q1_answer), ('Q2', 'no'), ('Q3', 'yes')]
    if gold == 'AFTER':
        return [('Q1', q1_answer), ('Q2', 'no'), ('Q3', 'no'), ('Q4', 'yes')]
    if gold == 'VAGUE':
        return [('Q1', q1_answer), ('Q2', 'no'), ('Q3', 'no'), ('Q4', 'no')]
    return None


def implied_label_from_q234(turns):
    """Walk Q2/Q3/Q4 in turns (turns[1:]) and return the Yuan-tree-implied label.
    Returns None if any required answer is uncertain/missing.
    """
    qids = ['Q1', 'Q2', 'Q3', 'Q4']
    answers = {}
    for i, t in enumerate(turns):
        if i >= len(qids):
            return None
        p = t.get('parsed')
        if p not in ('yes', 'no'):
            return None
        answers[qids[i]] = p
    q2 = answers.get('Q2')
    q3 = answers.get('Q3')
    q4 = answers.get('Q4')
    if q2 == 'yes':
        return 'EQUAL'
    if q2 != 'no':
        return None
    if q3 == 'yes':
        return 'BEFORE'
    if q3 != 'no':
        return None
    if q4 == 'yes':
        return 'AFTER'
    if q4 == 'no':
        return 'VAGUE'
    return None


def chain_from_teacher(turns):
    """Return list of (qid, parsed) for teacher trace; None if any uncertain.

    Walks Q1, Q2, Q3, Q4 in order. Pairs always start with Q1.
    """
    qids = ['Q1', 'Q2', 'Q3', 'Q4']
    chain = []
    for i, t in enumerate(turns):
        if i >= len(qids):
            return None  # too many turns
        p = t.get('parsed')
        if p not in ('yes', 'no'):
            return None
        chain.append((qids[i], p))
    return chain


def chain_terminal_label(chain):
    """Verify the chain terminates correctly per Yuan tree; return label or None."""
    # Q1's answer is routing only; Q2/Q3/Q4 commit
    answers = {qid: ans for qid, ans in chain}
    n = len(chain)
    if n == 2:  # must be Q1+Q2
        if answers.get('Q2') == 'yes':
            return 'EQUAL'
        return None  # Q2=No should not terminate at 2 turns
    if n == 3:  # Q1+Q2+Q3
        if answers.get('Q2') == 'no' and answers.get('Q3') == 'yes':
            return 'BEFORE'
        return None
    if n == 4:  # Q1+Q2+Q3+Q4
        if answers.get('Q2') == 'no' and answers.get('Q3') == 'no':
            return 'AFTER' if answers.get('Q4') == 'yes' else 'VAGUE'
        return None
    return None


def extract_think(response):
    """Return interior of last <think>...</think> block (case-insensitive), or None."""
    m = list(THINK_RE.finditer(response or ''))
    if not m:
        return None
    return m[-1].group(1).strip()


def build_messages_no_think(chain, q1_text, e1_ref, e2_ref):
    """Build messages list for v4a/v4b: bare questions + Yes/No assistant."""
    messages = []
    q1_yes = (chain[0][1] == 'yes')
    for i, (qid, ans) in enumerate(chain):
        if qid == 'Q1':
            user = q1_text
        elif qid == 'Q2':
            user = construct_q2(e1_ref, e2_ref, in_event=q1_yes)
        elif qid == 'Q3':
            user = construct_q3(e1_ref, e2_ref, in_event=q1_yes)
        elif qid == 'Q4':
            user = construct_q4(e1_ref, e2_ref, in_event=q1_yes)
        messages.append({'role': 'user', 'content': user})
        messages.append({'role': 'assistant', 'content': yes_no_str(ans)})
    return messages


def assistant_terminal(content):
    """Extract terminal Yes/No from assistant content, ignoring any <think> block."""
    # Find tail after </Think> (capital T, our chosen sentinel) or </think>
    tail = content
    for tag in ('</Think>', '</think>'):
        if tag in tail:
            tail = tail.rsplit(tag, 1)[-1]
            break
    tail = tail.strip().lower()
    if tail.startswith('yes'):
        return 'yes'
    if tail.startswith('no'):
        return 'no'
    return None


def verify_q1yes_phrasing(messages):
    """Return True if Q1=Yes path uses 'in that event' for all subsequent question turns."""
    if not messages or messages[0]['role'] != 'user':
        return True
    if assistant_terminal(messages[1]['content']) != 'yes':
        return True  # Q1=No, no constraint
    for i in range(2, len(messages), 2):
        if 'in that event' not in messages[i]['content']:
            return False
    return True


def main():
    random.seed(42)
    print('loading docs...', flush=True)
    docs = load_doc_index()
    print(f'  {len(docs)} train docs', flush=True)

    # Load all traces
    traces = []
    with open(TRACES) as f:
        for line in f:
            traces.append(json.loads(line))
    print(f'loaded {len(traces)} trace rows', flush=True)

    skipped = {
        'v4a': Counter(),
        'v4b': Counter(),
        'v4c': Counter(),
    }
    rows_v4a = []
    rows_v4b = []
    rows_v4c = []
    v4c_q1yes_phrasing_mismatches = 0

    for tr in traces:
        doc_id = tr['doc_id']
        e1_id = tr['e1_id']
        e2_id = tr['e2_id']
        e1_trigger = tr['e1_trigger']
        e2_trigger = tr['e2_trigger']
        gold = tr['gold_label']
        pred = tr['predicted_label']
        turns = tr['turns']

        # Resolve doc + mentions
        doc = docs.get(doc_id)
        if doc is None:
            for s in skipped.values():
                s['missing_doc'] += 1
            continue
        m1 = doc['mentions'].get(str(e1_id))
        m2 = doc['mentions'].get(str(e2_id))
        if m1 is None or m2 is None:
            for s in skipped.values():
                s['missing_mention'] += 1
            continue
        marked_doc = mark_target_pair_in_doc(
            doc['tokens'], m1['tokens_ids'], m2['tokens_ids'], e1_id, e2_id
        )
        e1_ref = ref(e1_trigger, e1_id)
        e2_ref = ref(e2_trigger, e2_id)
        q1_text = construct_q1(marked_doc, e1_ref, e2_ref)

        is_correct = (pred == gold)

        # Teacher chain (for v4b/v4c and v4a-teacher path)
        teacher_chain = chain_from_teacher(turns)
        teacher_chain_label = chain_terminal_label(teacher_chain) if teacher_chain else None

        # ---------- v4a ----------
        if is_correct:
            if teacher_chain is None:
                skipped['v4a']['uncertain_or_bad_turns'] += 1
            elif teacher_chain_label != gold:
                skipped['v4a']['teacher_chain_mismatches_gold'] += 1
            else:
                msgs = build_messages_no_think(teacher_chain, q1_text, e1_ref, e2_ref)
                rows_v4a.append({
                    'messages': msgs,
                    'doc_id': doc_id, 'e1_id': e1_id, 'e2_id': e2_id,
                    'gold_label': gold, 'source': 'teacher-correct',
                    'terminal_q': teacher_chain[-1][0],
                })
        else:
            # teacher-wrong: Q1 always from teacher; Q2/Q3/Q4 from teacher if their
            # implied label matches gold, else gold-derived.
            t0_parsed = turns[0].get('parsed') if turns else None
            if t0_parsed not in ('yes', 'no'):
                skipped['v4a']['q1_uncertain'] += 1
            else:
                implied = implied_label_from_q234(turns)
                if implied == gold and teacher_chain is not None:
                    # teacher-correct-chain: emission flipped but chain was right
                    msgs = build_messages_no_think(teacher_chain, q1_text, e1_ref, e2_ref)
                    rows_v4a.append({
                        'messages': msgs,
                        'doc_id': doc_id, 'e1_id': e1_id, 'e2_id': e2_id,
                        'gold_label': gold, 'source': 'teacher-correct-chain',
                        'terminal_q': teacher_chain[-1][0],
                    })
                else:
                    # gold-derived: keep teacher's Q1, walk Q2/Q3/Q4 from gold
                    chain = chain_for_gold(gold, q1_answer=t0_parsed)
                    if chain is None:
                        skipped['v4a']['unknown_gold'] += 1
                    else:
                        msgs = build_messages_no_think(chain, q1_text, e1_ref, e2_ref)
                        rows_v4a.append({
                            'messages': msgs,
                            'doc_id': doc_id, 'e1_id': e1_id, 'e2_id': e2_id,
                            'gold_label': gold, 'source': 'gold-derived',
                            'terminal_q': chain[-1][0],
                        })

        # ---------- v4b (teacher-correct only, no think) ----------
        if is_correct:
            if teacher_chain is None:
                skipped['v4b']['uncertain_or_bad_turns'] += 1
            elif teacher_chain_label != gold:
                skipped['v4b']['teacher_chain_mismatches_gold'] += 1
            else:
                msgs = build_messages_no_think(teacher_chain, q1_text, e1_ref, e2_ref)
                rows_v4b.append({
                    'messages': msgs,
                    'doc_id': doc_id, 'e1_id': e1_id, 'e2_id': e2_id,
                    'gold_label': gold,
                    'terminal_q': teacher_chain[-1][0],
                })
        else:
            skipped['v4b']['teacher_wrong'] += 1

        # ---------- v4c (teacher-correct only, with think) ----------
        if is_correct:
            if teacher_chain is None:
                skipped['v4c']['uncertain_or_bad_turns'] += 1
            elif teacher_chain_label != gold:
                skipped['v4c']['teacher_chain_mismatches_gold'] += 1
            else:
                # Build think-augmented messages from teacher's actual question text
                msgs = []
                bad = False
                q1_yes = (teacher_chain[0][1] == 'yes')
                for i, (qid, ans) in enumerate(teacher_chain):
                    t = turns[i]
                    q_text = strip_keep_short(t.get('question', ''))
                    # Phrasing-rule check (we keep teacher's text either way; just log)
                    if qid != 'Q1' and q1_yes and ('in that event' not in q_text):
                        nonlocal_mismatches[0] += 1
                    think = extract_think(t.get('response', ''))
                    if think is None:
                        bad = True
                        break
                    msgs.append({'role': 'user', 'content': q_text})
                    msgs.append({
                        'role': 'assistant',
                        'content': f'<think>\n{think}\n</Think>\n\n{yes_no_str(ans)}',
                    })
                if bad:
                    skipped['v4c']['missing_think'] += 1
                else:
                    rows_v4c.append({
                        'messages': msgs,
                        'doc_id': doc_id, 'e1_id': e1_id, 'e2_id': e2_id,
                        'gold_label': gold,
                        'terminal_q': teacher_chain[-1][0],
                    })
        else:
            skipped['v4c']['teacher_wrong'] += 1

    # Write all three files (messages-only for Together; diagnostics in .meta.jsonl sidecar)
    for path, rows in [(OUT_V4A, rows_v4a), (OUT_V4B, rows_v4b), (OUT_V4C, rows_v4c)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for r in rows:
                f.write(json.dumps({'messages': r['messages']}) + '\n')
        meta_path = path.replace('.jsonl', '.meta.jsonl')
        with open(meta_path, 'w') as f:
            for r in rows:
                meta = {k: v for k, v in r.items() if k != 'messages'}
                f.write(json.dumps(meta) + '\n')
        print(f'wrote {len(rows)} rows -> {path} (+ {meta_path})', flush=True)

    print()
    return rows_v4a, rows_v4b, rows_v4c, skipped, v4c_q1yes_phrasing_mismatches


# Hack: nonlocal in main() for v4c_q1yes_phrasing_mismatches via list-as-cell
nonlocal_mismatches = [0]


def report(name, path, rows, skipped, extra=None):
    print(f'\n========== {name} ({path}) ==========')
    print(f'  rows generated:       {len(rows)}')
    src_pairs = len(rows)  # one row per pair
    print(f'  source pairs included: {src_pairs}')

    terminal_q = Counter(r['terminal_q'] for r in rows)
    print(f'  terminal-Q distribution:  {dict(sorted(terminal_q.items()))}')

    label_dist = Counter(r['gold_label'] for r in rows)
    print(f'  gold label distribution:  {dict(sorted(label_dist.items()))}')

    q1yes = sum(1 for r in rows
                if r['messages'] and assistant_terminal(r['messages'][1]['content']) == 'yes')
    print(f'  Q1=Yes: {q1yes}/{len(rows)} ({100*q1yes/max(1,len(rows)):.2f}%)')

    if extra:
        for k, v in extra.items():
            print(f'  {k}: {v}')

    # Sanity check: verify Q1=Yes rows use "in that event" for subsequent turns
    bad = 0
    for r in rows:
        if not verify_q1yes_phrasing(r['messages']):
            bad += 1
    print(f'  Q1=Yes phrasing-rule violations: {bad}')

    # Skipped breakdown
    print(f'  skipped pairs by reason: {dict(skipped)}')


def length_stats(name, rows):
    """Approximate token = chars/4 for full message content per row."""
    char_lens = []
    for r in rows:
        total_chars = sum(len(m['content']) for m in r['messages'])
        char_lens.append(total_chars)
    if not char_lens:
        print(f'\n[{name}] no rows')
        return
    tok = sorted(c / 4 for c in char_lens)
    n = len(tok)
    p = lambda q: tok[min(int(q * n), n - 1)]
    print(f'\n[{name}] approx-tokens (chars/4) over {n} rows:')
    print(f'  median: {p(0.50):.0f}')
    print(f'  p95:    {p(0.95):.0f}')
    print(f'  p99:    {p(0.99):.0f}')
    print(f'  max:    {tok[-1]:.0f}')


def sample_dump(name, rows, want_q1yes=False, k=2):
    """Pretty-print k random rows, optionally ensuring at least 1 has Q1=Yes."""
    print(f'\n---- random samples from {name} ----')
    if not rows:
        print('  (no rows)')
        return
    chosen = []
    if want_q1yes:
        q1yes_rows = [r for r in rows
                      if r['messages'] and assistant_terminal(r['messages'][1]['content']) == 'yes']
        if q1yes_rows:
            chosen.append(random.choice(q1yes_rows))
    while len(chosen) < k:
        r = random.choice(rows)
        if r in chosen:
            continue
        chosen.append(r)
    for i, r in enumerate(chosen[:k]):
        print(f'\n--- sample {i} (gold={r["gold_label"]} terminal={r["terminal_q"]}) ---')
        # Truncate doc in Q1 for readability
        msgs_print = []
        for m in r['messages']:
            c = m['content']
            if len(c) > 1500:
                c = c[:600] + f'\n... [{len(c)-1200} chars elided] ...\n' + c[-600:]
            msgs_print.append({'role': m['role'], 'content': c})
        print(json.dumps({**{k: v for k, v in r.items() if k != 'messages'},
                          'messages': msgs_print}, indent=2))


def together_check(path):
    print(f'\n---- together files check {path} ----')
    try:
        out = subprocess.run(
            ['.venv/bin/together', 'files', 'check', path],
            capture_output=True, text=True, timeout=120
        )
        print('STDOUT:', out.stdout)
        if out.stderr:
            print('STDERR:', out.stderr, file=sys.stderr)
    except Exception as e:
        print(f'together CLI error: {type(e).__name__}: {e}')


V4D_SRC = 'output/matres_train_v4d_full_reasoning_goldcond.jsonl'
V4D_OUT = 'output/matres_train_v4d_sft_format.jsonl'
HINT_RE = re.compile(r'\bhint\b', re.IGNORECASE)


def build_v4d():
    """Convert merged v4d raw file into Together SFT chat format.

    Schema of source rows: doc_id, e1_id, e2_id, gold_label, predicted_label, turns,
    generation_mode, [v2 metadata for gold-conditioned rows].
    Each turn has: question (bare), think, response, parsed.

    Output schema: {"messages": [{role, content}, ...]}
    Assistant content format: f"<think>\\n{think}\\n</Think>\\n\\n{Yes|No}"
    (uses </Think> capital T per R1-Distill chat-template requirement.)
    """
    if not os.path.exists(V4D_SRC):
        raise SystemExit(f'Missing source: {V4D_SRC}')

    raw = [json.loads(l) for l in open(V4D_SRC)]
    print(f'loaded {len(raw)} v4d raw rows')

    sft_rows = []
    skipped = Counter()
    for r in raw:
        turns = r.get('turns') or []
        if not turns:
            skipped['no_turns'] += 1
            continue
        # Every turn must have definitive parsed in {yes, no}
        if not all(t.get('parsed') in ('yes', 'no') for t in turns):
            skipped['uncertain_turn'] += 1
            continue
        # Walk must terminate cleanly (Q1+Q2 / Q1+Q2+Q3 / Q1+Q2+Q3+Q4)
        chain = chain_from_teacher(turns)
        if chain is None:
            skipped['bad_chain'] += 1
            continue
        terminal = chain_terminal_label(chain)
        if terminal != r['gold_label']:
            skipped['terminal_mismatches_gold'] += 1
            continue

        msgs = []
        for t in turns:
            q_text = strip_keep_short(t.get('question', ''))
            think = (t.get('think') or '').strip()
            ans = yes_no_str(t['parsed'])
            msgs.append({'role': 'user', 'content': q_text})
            msgs.append({
                'role': 'assistant',
                'content': f'<think>\n{think}\n</Think>\n\n{ans}',
            })

        sft_rows.append({
            'messages': msgs,
            'doc_id': r['doc_id'], 'e1_id': r['e1_id'], 'e2_id': r['e2_id'],
            'gold_label': r['gold_label'],
            'generation_mode': r.get('generation_mode'),
            'terminal_q': chain[-1][0],
        })

    # Verify no \bhint\b in ASSISTANT content (training target). User content can
    # legitimately contain "hint" because source documents (e.g. wsj_0584) may
    # use the word as English — that's input, not what the model is trained to emit.
    leak_count = 0
    leak_examples = []
    user_hint_count = 0
    for r in sft_rows:
        for m in r['messages']:
            if not HINT_RE.search(m['content']):
                continue
            if m['role'] == 'user':
                user_hint_count += 1
                continue
            leak_count += 1
            if len(leak_examples) < 3:
                idx = HINT_RE.search(m['content']).start()
                leak_examples.append({
                    'doc_id': r['doc_id'], 'pair': (r['e1_id'], r['e2_id']),
                    'role': m['role'],
                    'ctx': m['content'][max(0, idx-60):idx+60],
                })
    print(f'\\bhint\\b in user content (allowed; source-doc English): {user_hint_count}')
    print(f'\\bhint\\b in assistant content: {leak_count}  (must be 0)')
    if leak_count:
        for ex in leak_examples:
            print(f'   {ex["doc_id"]} pair={ex["pair"]} role={ex["role"]}: ...{ex["ctx"]!r}...')
        raise SystemExit(1)

    os.makedirs(os.path.dirname(V4D_OUT), exist_ok=True)
    with open(V4D_OUT, 'w') as f:
        for r in sft_rows:
            f.write(json.dumps({'messages': r['messages']}) + '\n')
    meta_path = V4D_OUT.replace('.jsonl', '.meta.jsonl')
    with open(meta_path, 'w') as f:
        for r in sft_rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'messages'}) + '\n')

    print(f'\nwrote {len(sft_rows)} rows -> {V4D_OUT} (+ {meta_path})')
    print(f'skipped: {dict(skipped)}')

    return sft_rows


def report_v4d(rows):
    print(f'\n========== v4d (full+goldcond, with-think) ==========')
    print(f'  rows: {len(rows)}')

    label_dist = Counter(r['gold_label'] for r in rows)
    print(f'  gold label dist:   {dict(sorted(label_dist.items()))}')

    mode_dist = Counter(r['generation_mode'] for r in rows)
    print(f'  generation_mode:   {dict(mode_dist.most_common())}')

    terminal_q = Counter(r['terminal_q'] for r in rows)
    print(f'  terminal-Q:        {dict(sorted(terminal_q.items()))}')

    q1yes = sum(1 for r in rows
                if r['messages'] and assistant_terminal(r['messages'][1]['content']) == 'yes')
    print(f'  Q1=Yes: {q1yes}/{len(rows)} ({100*q1yes/max(1,len(rows)):.2f}%)')

    bad = sum(1 for r in rows if not verify_q1yes_phrasing(r['messages']))
    print(f'  Q1=Yes phrasing-rule violations: {bad}')

    # Per-class × mode
    xtab = Counter()
    for r in rows:
        xtab[(r['gold_label'], r['generation_mode'])] += 1
    print(f'\n  class × generation_mode:')
    for c in sorted(label_dist):
        fwd = xtab[(c, 'forward_pass')]
        gc  = xtab[(c, 'gold_conditioned_v2_per_turn_hint')]
        print(f'    {c:7}: forward_pass={fwd:5}  gold_cond={gc:5}  total={fwd+gc:5}')


def sample_dump_v4d(rows):
    """3 samples: 1 EQUAL, 1 VAGUE, 1 BEFORE/AFTER. One from each generation_mode at minimum."""
    print(f'\n---- v4d samples (1 EQUAL, 1 VAGUE, 1 BEFORE/AFTER) ----')

    by_label = {}
    for r in rows:
        by_label.setdefault(r['gold_label'], []).append(r)

    chosen = []
    for label in ('EQUAL', 'VAGUE', 'BEFORE', 'AFTER'):
        if label in by_label and len(chosen) < 3:
            chosen.append(random.choice(by_label[label]))
            if label == 'BEFORE' or label == 'AFTER':
                break  # pick only one of BEFORE/AFTER

    for i, r in enumerate(chosen):
        print(f'\n--- sample {i} (gold={r["gold_label"]} mode={r["generation_mode"]} terminal={r["terminal_q"]}) ---')
        msgs_print = []
        for m in r['messages']:
            c = m['content']
            if len(c) > 1500:
                c = c[:600] + f'\n... [{len(c)-1200} chars elided] ...\n' + c[-600:]
            msgs_print.append({'role': m['role'], 'content': c})
        print(json.dumps({'doc_id': r['doc_id'], 'pair': (r['e1_id'], r['e2_id']),
                          'gold_label': r['gold_label'],
                          'generation_mode': r['generation_mode'],
                          'terminal_q': r['terminal_q'],
                          'messages': msgs_print}, indent=2))


if __name__ == '__main__':
    if '--v4d' in sys.argv:
        random.seed(42)
        rows = build_v4d()
        print('\n\n############# REPORT #############')
        report_v4d(rows)

        print('\n\n############# LENGTH STATS #############')
        length_stats('v4d', rows)

        print('\n\n############# SAMPLES #############')
        sample_dump_v4d(rows)

        print('\n\n############# TOGETHER CHECK #############')
        together_check(V4D_OUT)
        sys.exit(0)

    rows_v4a, rows_v4b, rows_v4c, skipped, _ = main()
    print('\n\n############# REPORTS #############')
    report('v4a (full, direct labels)', OUT_V4A, rows_v4a, skipped['v4a'],
           extra={'source breakdown': dict(Counter(r['source'] for r in rows_v4a))})
    report('v4b (subset, no-think)', OUT_V4B, rows_v4b, skipped['v4b'])
    report('v4c (subset, with-think)', OUT_V4C, rows_v4c, skipped['v4c'],
           extra={'Q1=Yes phrasing mismatches in trace text (kept teacher phrasing)': nonlocal_mismatches[0]})

    print('\n\n############# LENGTH STATS #############')
    length_stats('v4a', rows_v4a)
    length_stats('v4b', rows_v4b)
    length_stats('v4c', rows_v4c)

    print('\n\n############# SAMPLES #############')
    sample_dump('v4a', rows_v4a)
    sample_dump('v4b', rows_v4b)
    sample_dump('v4c', rows_v4c, want_q1yes=True)

    print('\n\n############# TOGETHER CHECK #############')
    together_check(OUT_V4A)
    together_check(OUT_V4B)
    together_check(OUT_V4C)
