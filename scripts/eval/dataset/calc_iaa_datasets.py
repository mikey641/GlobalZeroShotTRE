# load json file
import json
import os

from sklearn.metrics import cohen_kappa_score

from scripts.eval.dataset.annot_obj import AnnotObj
from scripts.eval.dataset.iaa_result_obj import IAAResultObj
from scripts.eval.dataset.utils import count_stats_in_file, get_annotations, find_diffs


def adjust_relation(pairs):
    adjusted_pairs = list()
    for p in pairs:
        if p[2] == 'uncertain' or p[2] == 'vague':
            # pass
            adjusted_pairs.append((p[0], p[1], 'vague'))
        elif p[2] == 'includes':
            adjusted_pairs.append((p[0], p[1], 'before'))
        elif p[2] == 'is_included':
            adjusted_pairs.append((p[0], p[1], 'after'))
        else:
            adjusted_pairs.append(p)

    return adjusted_pairs


def calculate_iaa(ments1, pairs1, ments2, pairs2):
    tmp_annot1, _, _ = get_annotations(pairs1)
    tmp_annot2, _, _ = get_annotations(pairs2)

    tmp_annot1 = adjust_relation(tmp_annot1)
    tmp_annot2 = adjust_relation(tmp_annot2)

    joint_pairs = list(set([(p1[0], p1[1]) for p1 in tmp_annot1]).intersection(set([(p2[0], p2[1]) for p2 in tmp_annot2])))

    only_relv_pairs1 = sorted([p for p in tmp_annot1 if (p[0], p[1]) in joint_pairs])
    only_relv_pairs2 = sorted([p for p in tmp_annot2 if (p[0], p[1]) in joint_pairs])

    tmp_annot1_unpacked = [item[2] for item in only_relv_pairs1]
    tmp_annot2_unpacked = [item[2] for item in only_relv_pairs2]

    return tmp_annot1_unpacked, tmp_annot2_unpacked, only_relv_pairs1, only_relv_pairs2


def validate_mentions(mentions1, mentions2):
    mentions1.sort(key=lambda x: x['m_id'])
    mentions2.sort(key=lambda x: x['m_id'])
    for m1, m2 in zip(mentions1, mentions2):
        if m1['m_id'] == m2['m_id']:
            assert m1['tokens'] == m2['tokens']


def main(annot1_file, annot2_file):
    with open(annot1_file) as f:
        data1 = json.load(f)

    with open(annot2_file) as f:
        data2 = json.load(f)

    mentions1, num_tmp_pairs1, num_equal_pairs1, num_before_after_pairs1, _, expected_pairs1 = count_stats_in_file(data1)
    mentions2, num_tmp_pairs2, num_equal_pairs2, num_before_after_pairs2, _, expected_pairs2 = count_stats_in_file(data2)

    joint_mentions_ids = list(set([m1['m_id'] for m1 in mentions1]).intersection(set([m2['m_id'] for m2 in mentions2])))
    mentions1 = [m for m in mentions1 if m['m_id'] in joint_mentions_ids]
    mentions2 = [m for m in mentions2 if m['m_id'] in joint_mentions_ids]

    validate_mentions(mentions1, mentions2)

    tmp_annot1_unpacked, tmp_annot2_unpacked, only_relv_pairs1, only_relv_pairs2 = calculate_iaa(mentions1, data1["allPairs"], mentions2, data2["allPairs"])

    return tmp_annot1_unpacked, tmp_annot2_unpacked, only_relv_pairs1, only_relv_pairs2, mentions1


def compute_iaa(group_tmp_results):
    total_ment = sum([len(iaa_obj[0].mentions) for iaa_obj in group_tmp_results.values()])
    total_tmp_pairs = sum([iaa_obj[0].num_tmp_pairs for iaa_obj in group_tmp_results.values()])
    total_diff = sum([len(iaa_obj[2].diff) for iaa_obj in group_tmp_results.values()])
    total_same = sum([len(iaa_obj[2].same) for iaa_obj in group_tmp_results.values()])
    avg_agreement_tmp = sum([iaa_obj[2].iaa for iaa_obj in group_tmp_results.values()]) / len(group_tmp_results)

    total_diff_same = total_diff + total_same

    print(f'Total-Files: {len(group_tmp_results.keys())}')
    print(f'Total-mentions: {total_ment}')
    print(f'Total-Temp-pairs: {total_tmp_pairs}')
    print(f'Avg-Temp-pairs: {total_tmp_pairs/len(group_tmp_results)}')
    print(f'Total-disagree: {total_diff}')
    print(f'Total-agree: {total_same}')
    print(f'Total-(diff + same): {total_diff_same}')
    print(f'Average tmp agreement: {avg_agreement_tmp}')
    print('---------------------------------')


if __name__ == "__main__":
    annotator1 = 'MATRES'
    annotator2 = 'TBD'
    _folder1 = 'data/MATRES/in_my_format/all'
    _folder2 = 'data/TimeBank-Dense/all_converted'

    _group_tmp_results = dict()
    _all_files = os.listdir(_folder1)
    _tmp_annot1_all = list()
    _tmp_annot2_all = list()
    _diffs_all = dict()
    for file in os.listdir(_folder2):
        if file in _all_files:
            _annot1_file = f'{_folder1}/{file}'
            _annot2_file = f'{_folder2}/{file}'
            _tmp_annot1_unpacked, _tmp_annot2_unpacked, _only_relv_pairs1, _only_relv_pairs2, _mentions1 = main(_annot1_file, _annot2_file)
            _tmp_annot1_all.extend(_tmp_annot1_unpacked)
            _tmp_annot2_all.extend(_tmp_annot2_unpacked)

            tmp_diff, tmp_sames, tmp_unagreed = find_diffs(_mentions1, _only_relv_pairs1, _only_relv_pairs2)
            _diffs_all[file] = (tmp_diff, tmp_sames, tmp_unagreed)
            # _group_tmp_results[file] = (_annot_obj1, _annot_obj2, _tmp_iaa_result)

    tmp_score = cohen_kappa_score(_tmp_annot1_all, _tmp_annot2_all)

    print(f'Average tmp agreement: {tmp_score}')
    print(f'Total-Files: {len(_diffs_all.keys())}')
    print(f'Total-diffs: {sum([len(diff[0]) for diff in _diffs_all.values()])}')
    print(f'Total-sames: {sum([len(diff[1]) for diff in _diffs_all.values()])}')
    print(f'Total-unagreed: {sum([len(diff[2]) for diff in _diffs_all.values()])}')

    # create_report(_annot_obj1, _annot_obj2, _tmp_iaa_result, _coref_iaa_result, _cause_iaa_result)
    print("Done!")
