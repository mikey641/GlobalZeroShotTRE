import json
import os
from collections import Counter

import spacy

from scripts.data_process.my_format_sliding_window import process_doc

COMPOSITION = {
    'B': {  # "before"
        'B': 'B',
        'A': 'V',
        'I': 'V' ,
        'II': 'V',
        'E': 'B',
        'V': 'V'
    },
    'A': {  # "after"
        'B': 'V',
        'A': 'A',
        'I': 'V',
        'II': 'V',
        'E': 'A',
        'V': 'V'
    },
    'I': {  # "includes"
        'B': 'V',
        'A': 'V',
        'I': 'I',
        'II': 'V',
        'E': 'I',
        'V': 'V'
    },
    'II': {  # "is-included"
        'B': 'V',
        'A': 'V',
        'I': 'V',
        'II': 'II',
        'E': 'II',
        'V': 'V'
    },
    'E': {  # "equal"
        'B': 'B',
        'A': 'A',
        'I': 'I',
        'II': 'II',
        'E': 'E',
        'V': 'V'
    },
    'V': {  # "vague"
        'B': 'V',
        'A': 'V',
        'I': 'V',
        'II': 'V',
        'E': 'V',
        'V': 'V'
    }
}


def convert_rel(rel):
    if rel == 'before':
        return 'B'
    elif rel == 'after':
        return 'A'
    elif rel == 'includes':
        return 'I'
    elif rel == 'is_included':
        return 'II'
    elif rel == 'equal':
        return 'E'
    elif rel == 'vague':
        return 'V'
    else:
        return None


def convert_pairs(all_pairs):
    all_relations = dict()
    mentions = set()
    for pair in all_pairs:
        all_relations[(pair['_firstId'], pair['_secondId'])] = convert_rel(pair['_relation'])
        mentions.add(pair['_firstId'])
        mentions.add(pair['_secondId'])
    return all_relations, list(mentions)


def compute_transitive_closure(nodes, relations):
    """
    nodes: list of all entities (events/intervals)
    relations: dictionary keyed by (i, j) -> relation (str or set)

    Returns: a possibly expanded 'relations' dict with inferred relations.
    """
    changed = True

    while changed:
        changed = False
        for i in nodes:
            for j in nodes:
                if (i, j) not in relations:
                    continue

                r_ij = relations[(i, j)]

                for k in nodes:
                    if (j, k) not in relations or (i, k) in relations:
                        continue

                    r_jk = relations[(j, k)]
                    new_r_ik = COMPOSITION[r_ij][r_jk]

                    if (i, k) not in relations:
                        relations[(i, k)] = new_r_ik
                        changed = True


def get_file_pairs(input_folder):
    sent_diff = Counter()
    all_pairs_orig = dict()
    all_doc_json_mentions = dict()
    all_rels_count = Counter()
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_folder, file_name)
            with_sent_dist = process_doc(input_file, _nlp, only_consecutive=False)
            json_file_data = json.load(open(input_file))

            assert with_sent_dist['tokens'] == json_file_data['tokens']
            assert len(with_sent_dist['allMentions']) == len(json_file_data['allMentions'])
            assert len(with_sent_dist['allPairs']) == len(json_file_data['allPairs'])

            all_pairs = with_sent_dist['allPairs']
            all_relations, _mentions = convert_pairs(all_pairs)
            all_pairs_orig[file_name] = (_mentions, all_relations)
            all_doc_json_mentions[file_name] = with_sent_dist['allMentions']
            all_rels_count.update(all_relations.values())

            for pair in all_pairs:
                if pair['_relation'] != 'vague':
                    sent_diff[pair['send_diff']] += 1

    print(f'Total relations: {sum(all_rels_count.values())}')
    print(f'Relations: {all_rels_count}')
    print('---')
    print(f'Total sentence differences: {sent_diff}')
    print('---')
    return all_pairs_orig, all_doc_json_mentions


def clear_symmetry_and_vague(relations):
    for (i, j) in list(relations.keys()):
        if relations[(i, j)] == 'V':
            del relations[(i, j)]


def create_sent_distrib(doc_ments, pairs):
    sent_diff = Counter()
    doc_id_to_ment = {ment['m_id']: ment for ment in doc_ments}
    for pair_tup, value in pairs.items():
        if value != 'V':
            i, j = pair_tup
            mi = doc_id_to_ment[i]
            mj = doc_id_to_ment[j]
            sent_diff[abs(mi['sent_id'] - mj['sent_id'])] += 1

    return sent_diff


if __name__ == "__main__":
    _nlp = spacy.load("en_core_web_trf")

    _input_folder_original = 'data/NarrativeTime/converted_no_overlap/test'
    _input_folder_cons_sents = 'data/NarrativeTime/converted_no_overlap/test_consecutive_sents'

    _all_pairs_orig, _all_doc_json_mentions = get_file_pairs(_input_folder_original)
    _all_pairs_cons_sent, _all_cons_json_mentions = get_file_pairs(_input_folder_cons_sents)

    for file, tup in _all_pairs_cons_sent.items():
        compute_transitive_closure(tup[0], tup[1])

    all_orig_sent_diff = Counter()
    all_cons_sent_diff = Counter()
    all_orig_after_count = Counter()
    all_cons_after_count = Counter()
    for file, tup in _all_pairs_orig.items():
        _origin_pairs = tup[1]
        _const_trans_pairs = _all_pairs_cons_sent[file][1]
        # clear_symmetry_and_vague(origin_pairs)
        # clear_symmetry_and_vague(const_trans_pairs)
        all_orig_after_count.update(_origin_pairs.values())
        all_cons_after_count.update(_const_trans_pairs.values())

        all_orig_sent_diff.update(create_sent_distrib(_all_doc_json_mentions[file], _origin_pairs))
        all_cons_sent_diff.update(create_sent_distrib(_all_cons_json_mentions[file], _const_trans_pairs))

    print(f'Total original relations: {sum(all_orig_after_count.values())}')
    print(f'Original relations: {all_orig_after_count}')
    print('---')
    print(f'Total consecutive relations: {sum(all_cons_after_count.values())}')
    print(f'Consecutive relations: {all_cons_after_count}')
    print('---')
    print(f'Total original sentence differences: {all_orig_sent_diff}')
    print(f'Total consecutive sentence differences: {all_cons_sent_diff}')
    print('---')
