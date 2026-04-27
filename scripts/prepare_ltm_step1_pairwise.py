"""Build LTM Step 1 training file: pairwise prompt × all MATRES train pairs.

For each MATRES train doc, emit one chat row per pair:
  user      = build_pairwise_prompt(doc, pair)        (from src/matres_prompt.py)
  assistant = pair['_relation']                        (BEFORE / AFTER / EQUAL / VAGUE)

Output: output/matres_train_ltm_step1_pairwise.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

# Inline copy of matres_prompt.build_pairwise_prompt (from src/matres_prompt.py)
def mark_events_in_text(tokens, all_mentions):
    tokens = list(tokens)
    for mention in all_mentions:
        first = mention['tokens_ids'][0]
        last = mention['tokens_ids'][-1]
        tokens[first] = f'<{tokens[first]}'
        tokens[last] = f'{tokens[last]}(ei{mention["m_id"]})>'
    return ' '.join(tokens)


def build_pairwise_prompt(data, pair):
    text = mark_events_in_text(data['tokens'], data['allMentions'])
    mention_text = {str(m['m_id']): str(m['tokens']) for m in data['allMentions']}
    first_id = pair['_firstId']
    second_id = pair['_secondId']
    first = mention_text.get(str(first_id), str(first_id))
    second = mention_text.get(str(second_id), str(second_id))
    return (
        "\nGiven the text below where events are marked with <eventName(identifier)>,"
        " for the specified pair of events below, determine the temporal relationships "
        "(BEFORE, AFTER, EQUAL, VAGUE) between them.\n\nText -\n"
        f"{text}\n\n"
        "Pair -\n"
        f"{first}(ei{first_id}) -- {second}(ei{second_id})\n\n"
        "Answer -\n"
    )


TRAIN_FOLDER = 'data/MATRES/_in_OmniTemp_format/train'
OUT = 'output/matres_train_ltm_step1_pairwise.jsonl'


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_docs = 0
    n_rows = 0
    label_dist = Counter()

    with open(OUT, 'w') as g:
        for fn in sorted(os.listdir(TRAIN_FOLDER)):
            if not fn.endswith('.json'):
                continue
            n_docs += 1
            with open(os.path.join(TRAIN_FOLDER, fn)) as f:
                doc = json.load(f)
            for p in doc.get('allPairs', []):
                rel = p.get('_relation', '').strip().upper()
                if rel not in ('BEFORE', 'AFTER', 'EQUAL', 'VAGUE'):
                    continue
                user_text = build_pairwise_prompt(doc, p)
                row = {
                    'messages': [
                        {'role': 'user', 'content': user_text},
                        {'role': 'assistant', 'content': rel},
                    ]
                }
                g.write(json.dumps(row) + '\n')
                n_rows += 1
                label_dist[rel] += 1

    size = os.path.getsize(OUT)
    print(f'wrote {n_rows} rows from {n_docs} docs -> {OUT}')
    print(f'  size: {size:,} bytes ({size/1024/1024:.1f} MB)')
    print(f'  label dist: {dict(label_dist.most_common())}')


if __name__ == '__main__':
    main()
