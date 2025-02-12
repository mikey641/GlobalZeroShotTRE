import json
import os

from sklearn.metrics import cohen_kappa_score, precision_score, recall_score

from scripts.eval.dataset.calc_iaa_datasets import validate_mentions, calculate_iaa
from scripts.eval.dataset.utils import find_diffs, count_stats_in_file


ANNOT1_RELATION = 'vague'
ANNOT2_RELATION = 'uncertain'

# ANNOT1_RELATION = 'after'
# ANNOT2_RELATION = 'after'

def main(annot1_file, annot2_file):
    with open(annot1_file) as f:
        data1 = json.load(f)

    with open(annot2_file) as f:
        data2 = json.load(f)

    all_pairs_ = data1["allPairs"]
    mentions1 = data1["allMentions"]
    annot1_relations = []
    only_annot1_nodes_ids = set()
    for pair in all_pairs_:
        if pair['_relation'].lower() == ANNOT1_RELATION:
            annot1_relations.append(pair)
            only_annot1_nodes_ids.add(pair['_firstId'])
            only_annot1_nodes_ids.add(pair['_secondId'])

    annot1_mentions = [m for m in mentions1 if m['m_id'] in only_annot1_nodes_ids]
    annot2_mentions = [m for m in data2['allMentions'] if m['m_id'] in only_annot1_nodes_ids]
    joint_ment = list(set([m['m_id'] for m in annot1_mentions]).intersection(set([m['m_id'] for m in annot2_mentions])))
    annot1_pairs_ids = set([(p['_firstId'], p['_secondId']) for p in annot1_relations if p['_firstId'] in joint_ment and p['_secondId'] in joint_ment])
    annot1_pairs_ids.update([(p['_secondId'], p['_firstId']) for p in annot1_relations if p['_firstId'] in joint_ment and p['_secondId'] in joint_ment])
    annot2_pairs_ids = set([(p['_firstId'], p['_secondId']) for p in data2['allPairs'] if p['_firstId'] in joint_ment and p['_secondId'] in joint_ment])

    join_pair_ids = annot1_pairs_ids.intersection(annot2_pairs_ids)

    annot2_relations = [p for p in data2['allPairs'] if (p['_firstId'], p['_secondId']) in join_pair_ids or (p['_secondId'], p['_firstId']) in join_pair_ids]
    annot1_relations = [p for p in annot1_relations if (p['_firstId'], p['_secondId']) in join_pair_ids or (p['_secondId'], p['_firstId']) in join_pair_ids]

    validate_mentions(annot1_mentions, annot2_mentions)
    for rel in annot2_relations:
        if rel['_relation'].lower() == ANNOT2_RELATION:
            rel['_relation'] = ANNOT1_RELATION
        else:
            rel['_relation'] = 'not_compatible'

    annot1_relations.sort(key=lambda x: (x['_firstId'], x['_secondId']))
    annot2_relations.sort(key=lambda x: (x['_firstId'], x['_secondId']))
    return annot1_relations, annot2_relations


if __name__ == "__main__":
    annotator1 = 'MATRES'
    annotator2 = 'TBD'
    _folder1 = 'data/MATRES/in_my_format/all'
    _folder2 = 'data/TimeBank-Dense/all_converted'

    # annotator1 = 'TBD'
    # annotator2 = 'MATRES'
    # _folder1 = 'data/TimeBank-Dense/all_converted'
    # _folder2 = 'data/MATRES/in_my_format/all'

    _group_tmp_results = dict()
    _all_files = os.listdir(_folder1)
    _tmp_annot1_all = list()
    _tmp_annot2_all = list()
    _diffs_all = dict()
    shared_files = 0
    for file in os.listdir(_folder2):
        if file in _all_files:
            _annot1_file = f'{_folder1}/{file}'
            _annot2_file = f'{_folder2}/{file}'
            _only_matres_vague_relations, _tbd_relations = main(_annot1_file, _annot2_file)
            _tmp_annot1_all.extend(_only_matres_vague_relations)
            _tmp_annot2_all.extend(_tbd_relations)
            shared_files += 1

    print(f'Shared files: {shared_files}')
    _final_matres = [1 if pair['_relation'].lower() == ANNOT1_RELATION else 0 for pair in _tmp_annot1_all]
    _final_tbd = [1 if pair['_relation'].lower() == ANNOT1_RELATION else 0 for pair in _tmp_annot2_all]

    print(f'list_A = {_final_matres}')
    print(f'list_B = {_final_tbd}')

    recall = recall_score(_final_matres, _final_tbd)

    true_positives = sum(1 for x, y in zip(_final_matres, _final_tbd) if x == y == 1)
    agreement_ratio = true_positives / len(_final_matres)

    print(f'Total pairs: {len(_final_matres)}')
    print(f'True positives: {true_positives}')
    print(f'agreement: {agreement_ratio}')
    print(f'Recall: {recall}')

    # create_report(_annot_obj1, _annot_obj2, _tmp_iaa_result, _coref_iaa_result, _cause_iaa_result)
    print("Done!")
