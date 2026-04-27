"""Test 1 (corrected) + eval-time agreement + oracle F1.

Yuan chain-verdict mapping (Q1 ignored, it only affects phrasing):
    Q2=Yes               -> EQUAL
    Q3=Yes               -> BEFORE
    Q4=Yes               -> AFTER
    otherwise            -> VAGUE

Runs three reports:
  (A) training-data chain vs emitted-label agreement
      (output/matres_train_sft_format_v3.jsonl)
  (B) eval-trace chain vs emitted-label agreement
      (output/v3_epoch2_matres_test.traces.jsonl)
  (C) oracle F1 (chain-derived labels) vs emitted F1 on the eval traces,
      via the MATRES evaluation pipeline.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/test1_chain_agreement_v2.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.utils.classes.datasets_type import MatresDataset
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation


TRAIN_PATH = 'output/matres_train_sft_format_v3.jsonl'
EVAL_TRACES = 'output/v3_epoch2_matres_test.traces.jsonl'
ORACLE_DOT = 'output/v3_epoch2_matres_test_dot/matres_v3_epoch2_oracle.json'
EMITTED_DOT = 'output/v3_epoch2_matres_test_dot/matres_v3_epoch2_emitted_resliced.json'

Q1_RE = re.compile(r'Are <EVENT [^>]+>.*?</EVENT> and <EVENT [^>]+>.*?</EVENT> referring to the same event\?')
Q2_RE = re.compile(r'Did <EVENT [^>]+>.*?</EVENT> and <EVENT [^>]+>.*?</EVENT> simultaneously happen(?: in that event)?\?')
Q3_RE = re.compile(r'Is <EVENT [^>]+>.*?</EVENT> before <EVENT [^>]+>.*?</EVENT>(?: in that event)?\?')
Q4_RE = re.compile(r'Is <EVENT [^>]+>.*?</EVENT> after <EVENT [^>]+>.*?</EVENT>(?: in that event)?\?')

FINAL_LABEL_RE = re.compile(r'\b(BEFORE|AFTER|EQUAL|VAGUE)\b')


def split_questions(chain):
    markers = []
    for qid, pat in [('Q1', Q1_RE), ('Q2', Q2_RE), ('Q3', Q3_RE), ('Q4', Q4_RE)]:
        m = pat.search(chain)
        if m:
            markers.append((m.start(), m.end(), qid))
    markers.sort()
    blocks = {}
    for i, (s, e, qid) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(chain)
        blocks[qid] = chain[e:end]
    return blocks


def answer_of(block):
    """yes / no / uncertain / None. Prefer explicit 'So: Yes/No.' summary."""
    if block is None:
        return None
    m = re.search(r'\bSo:\s*(Yes|No)\b', block, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Fallback: look at the last non-empty line for Yes/No
    for line in reversed([ln.strip() for ln in block.splitlines() if ln.strip()]):
        low = line.lower()
        if low.startswith('yes'):
            return 'yes'
        if low.startswith('no'):
            return 'no'
        if re.search(r'\byes\b', low) and not re.search(r'\bno\b|\bnot\b', low):
            return 'yes'
        if re.search(r'\bno\b|\bnot\b', low) and not re.search(r'\byes\b', low):
            return 'no'
        break
    return 'uncertain'


def chain_verdict(chain):
    """Corrected Yuan mapping: Q2=Yes→EQUAL, Q3=Yes→BEFORE, Q4=Yes→AFTER, else VAGUE.
    Q1 is ignored for verdict (it only affects phrasing of Q2/Q3/Q4).
    """
    blocks = split_questions(chain)
    q2 = answer_of(blocks.get('Q2'))
    q3 = answer_of(blocks.get('Q3'))
    q4 = answer_of(blocks.get('Q4'))
    if q2 == 'yes':
        return 'EQUAL'
    if q3 == 'yes':
        return 'BEFORE'
    if q4 == 'yes':
        return 'AFTER'
    return 'VAGUE'


def emitted_label(content):
    if '</Think>' in content:
        tail = content.rsplit('</Think>', 1)[-1]
        m = FINAL_LABEL_RE.search(tail.upper())
        if m:
            return m.group(1)
    m = FINAL_LABEL_RE.search(content[-200:].upper())
    return m.group(1) if m else None


def chain_of(content):
    return content.split('</Think>', 1)[0] if '</Think>' in content else content


def assistant_content(row):
    for m in row.get('messages', []):
        if m['role'] == 'assistant':
            return m['content']
    return ''


def score_agreement(iterable_rows, content_fn, title):
    """iterable_rows yields rows; content_fn(row) -> full assistant text."""
    total = 0
    agree = 0
    disagree = 0
    missing_emitted = 0
    disagreement_pairs = Counter()
    per_emitted = {}
    chain_dist = Counter()
    emit_dist = Counter()

    for row in iterable_rows:
        total += 1
        content = content_fn(row)
        chain = chain_of(content)
        verdict = chain_verdict(chain)
        emitted = emitted_label(content)

        chain_dist[verdict] += 1
        emit_dist[emitted or 'NONE'] += 1

        if emitted is None:
            missing_emitted += 1
            continue
        per_emitted.setdefault(emitted, {'agree': 0, 'disagree': 0})
        if verdict == emitted:
            agree += 1
            per_emitted[emitted]['agree'] += 1
        else:
            disagree += 1
            per_emitted[emitted]['disagree'] += 1
            disagreement_pairs[(verdict, emitted)] += 1

    scored = agree + disagree
    print(f'\n======== {title} ========')
    print(f'total rows:         {total}')
    print(f'  missing emitted:  {missing_emitted}')
    print(f'  scored:           {scored}')
    if scored:
        print(f'agree:    {agree} ({100*agree/scored:.2f}%)')
        print(f'disagree: {disagree} ({100*disagree/scored:.2f}%)')
    print(f'chain-verdict dist: {dict(chain_dist)}')
    print(f'emitted dist:       {dict(emit_dist)}')
    print('disagreement pairs (chain_verdict -> emitted):')
    for (v, e), n in disagreement_pairs.most_common(20):
        print(f'  {v:7s} -> {e:7s}  {n}')
    print('per-emitted consistency:')
    for label in ('BEFORE', 'AFTER', 'EQUAL', 'VAGUE'):
        if label in per_emitted:
            s = per_emitted[label]
            tot = s['agree'] + s['disagree']
            print(f"  {label:7s}  {s['agree']}/{tot} = {100*s['agree']/tot:.2f}% consistent")


def build_dot(traces, label_fn, out_path):
    per_doc_edges = {}
    for t in traces:
        doc = t['doc_id']
        per_doc_edges.setdefault(doc, [])
        lab = label_fn(t)
        if lab is None:
            continue
        e1 = str(t['e1_trigger']).replace('"', '')
        e2 = str(t['e2_trigger']).replace('"', '')
        per_doc_edges[doc].append(
            f'"{e1}({t["e1_id"]})" -- "{e2}({t["e2_id"]})" [rel={lab.lower()}];'
        )
    dot_obj = {
        doc: {'target': 'strict graph {\n' + '\n'.join(edges) + '\n}'}
        for doc, edges in per_doc_edges.items()
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(dot_obj, f, indent=2)


def score_dot(dot_path, tag):
    ds = MatresDataset()
    test_as_dict, all_test_files = load_golds(ds.get_test_file(), ds.get_label_set())
    pred_as_dict, _ = read_pred_dot_file(dot_path, all_test_files, ds)
    all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(
        test_as_dict, pred_as_dict, ds.get_label_set()
    )
    print(f'\n-------- {tag} ({dot_path}) --------')
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, ds)
    print(f'NAs (convert_format default→BEFORE): {count_nas}')
    print(f'F1: {f1:.4f}')
    return f1


def main():
    # (A) training
    def iter_train():
        with open(TRAIN_PATH) as f:
            for line in f:
                yield json.loads(line)

    score_agreement(iter_train(), assistant_content, 'A. TRAINING chain vs emitted')

    # (B) eval traces
    def iter_eval():
        with open(EVAL_TRACES) as f:
            for line in f:
                yield json.loads(line)

    # For eval, content_fn returns the raw_output
    score_agreement(iter_eval(), lambda t: t['raw_output'], 'B. EVAL chain vs emitted')

    # (C) oracle F1 vs emitted F1 (resliced to same subset)
    traces = []
    with open(EVAL_TRACES) as f:
        for line in f:
            traces.append(json.loads(line))

    # Emitted DOT (only on this resliced subset — we don't use the full-run DOT)
    build_dot(traces, lambda t: emitted_label(t['raw_output']), EMITTED_DOT)
    # Oracle DOT — chain-derived verdict
    build_dot(traces, lambda t: chain_verdict(chain_of(t['raw_output'])), ORACLE_DOT)

    print('\n======== C. ORACLE F1 vs EMITTED F1 (resliced on current traces) ========')
    emitted_f1 = score_dot(EMITTED_DOT, 'EMITTED (as-is from student)')
    oracle_f1 = score_dot(ORACLE_DOT, 'ORACLE (chain-derived by corrected Yuan tree)')
    print(f'\nΔ (oracle − emitted): {100*(oracle_f1 - emitted_f1):+.2f} F1 pts')


if __name__ == '__main__':
    main()
