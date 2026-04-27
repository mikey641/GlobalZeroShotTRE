"""Build v4c with NATIVE Together reasoning field (no <think> tags in content).

Source: output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl
Filter: teacher-correct pairs only (predicted_label == gold_label).

Each multi-turn row uses Yuan tree structure (Q1, Q2, Q3, Q4 with "in that
event" routing on Q1=Yes). Each assistant turn:
  {"role": "assistant", "reasoning": "<chain>", "content": "Yes"|"No"}

Skip pairs where any turn has parsed not in {"yes","no"} or has no extractable
<think> block.

Output: output/matres_train_v4c_subset_native_reasoning.jsonl
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from scripts.run.prompts_cot_yuan import mark_target_pair_in_doc, ref


TRACES = 'output/matres_train_continue_full/matres_train_continue_DeepSeek-R1.traces.jsonl'
DOC_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'
OUT = 'output/matres_train_v4c_subset_native_reasoning.jsonl'

THINK_RE = re.compile(r'<think>(.*?)</think>', re.IGNORECASE | re.DOTALL)
KEEP_SHORT_RE = re.compile(r'\s*Keep the answer short and concise\.\s*$')


def strip_keep_short(s):
    return KEEP_SHORT_RE.sub('', s).rstrip()


def extract_think(response):
    if not response:
        return None
    m = list(THINK_RE.finditer(response))
    if not m:
        return None
    return m[-1].group(1).strip()


def load_doc_index():
    docs = {}
    for fn in sorted(os.listdir(DOC_FOLDER)):
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(DOC_FOLDER, fn)) as f:
            d = json.load(f)
        docs[fn] = {
            'tokens': d['tokens'],
            'mentions': {m['m_id']: m for m in d.get('allMentions', [])},
        }
    return docs


def chain_terminal_label(turns):
    """Return (label, terminal_qid) from teacher's parsed yes/no walk; or (None, None)."""
    qids = ['Q1', 'Q2', 'Q3', 'Q4']
    parsed = []
    for i, t in enumerate(turns):
        if i >= 4:
            return None, None
        p = t.get('parsed')
        if p not in ('yes', 'no'):
            return None, None
        parsed.append(p)
    n = len(parsed)
    if n == 2:
        if parsed[1] == 'yes':
            return 'EQUAL', 'Q2'
    elif n == 3:
        if parsed[1] == 'no' and parsed[2] == 'yes':
            return 'BEFORE', 'Q3'
    elif n == 4:
        if parsed[1] == 'no' and parsed[2] == 'no':
            return ('AFTER', 'Q4') if parsed[3] == 'yes' else ('VAGUE', 'Q4')
    return None, None


def main():
    docs = load_doc_index()
    print(f'loaded {len(docs)} train docs')

    rows = []
    skipped = Counter()
    label_dist = Counter()
    terminal_dist = Counter()
    q1yes = 0

    with open(TRACES) as f:
        for line in f:
            tr = json.loads(line)
            if tr['predicted_label'] != tr['gold_label']:
                skipped['teacher_wrong'] += 1
                continue

            label, terminal_q = chain_terminal_label(tr['turns'])
            if label is None:
                skipped['uncertain_or_bad_chain'] += 1
                continue
            if label != tr['gold_label']:
                # Defensive: should not happen for teacher-correct, but guard anyway
                skipped['chain_mismatch_gold'] += 1
                continue

            doc = docs.get(tr['doc_id'])
            if doc is None:
                skipped['missing_doc'] += 1
                continue
            m1 = doc['mentions'].get(str(tr['e1_id']))
            m2 = doc['mentions'].get(str(tr['e2_id']))
            if m1 is None or m2 is None:
                skipped['missing_mention'] += 1
                continue

            # Build messages — use teacher's actual question text (with "Keep…" stripped),
            # native reasoning field for each assistant turn.
            msgs = []
            bad = False
            for i, t in enumerate(tr['turns']):
                think = extract_think(t.get('response', ''))
                if think is None:
                    bad = True
                    break
                user_text = strip_keep_short(t.get('question', ''))
                msgs.append({'role': 'user', 'content': user_text})
                msgs.append({
                    'role': 'assistant',
                    'reasoning': think,
                    'content': 'Yes' if t['parsed'] == 'yes' else 'No',
                })
            if bad:
                skipped['missing_think'] += 1
                continue

            label_dist[label] += 1
            terminal_dist[terminal_q] += 1
            if tr['turns'][0].get('parsed') == 'yes':
                q1yes += 1

            rows.append({
                'messages': msgs,
                'doc_id': tr['doc_id'],
                'e1_id': tr['e1_id'], 'e2_id': tr['e2_id'],
                'gold_label': label,
                'terminal_q': terminal_q,
            })

    # Write messages-only training file + meta sidecar
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        for r in rows:
            f.write(json.dumps({'messages': r['messages']}) + '\n')
    meta = OUT.replace('.jsonl', '.meta.jsonl')
    with open(meta, 'w') as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'messages'}) + '\n')

    print()
    print(f'wrote {len(rows)} rows -> {OUT}')
    print(f'  + meta sidecar: {meta}')
    print()
    print('======== report ========')
    print(f'rows generated:           {len(rows)}')
    print(f'label distribution:       {dict(sorted(label_dist.items()))}')
    print(f'terminal-Q distribution:  {dict(sorted(terminal_dist.items()))}')
    print(f'Q1=Yes count:             {q1yes} ({100*q1yes/max(1,len(rows)):.2f}%)')
    print(f'source breakdown:         {{"teacher-correct": {len(rows)}}}')
    print(f'skipped:                  {dict(skipped)}')


if __name__ == '__main__':
    main()
