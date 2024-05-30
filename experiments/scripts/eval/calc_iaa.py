# load json file
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.eval.annot_obj import AnnotObj
from scripts.eval.iaa_result_obj import IAAResultObj
from scripts.eval.utils import count_stats_in_file, get_annotations, find_diffs


def create_report(annot_obj1, annot_obj2, tmp_iaa_result, coref_iaa_result, cause_iaa_result):
    df = pd.DataFrame(columns=['Mention1', 'Mention2', annotator1, annotator2], data=tmp_iaa_result.diff)
    df_string = df.to_string(index=False)
    file_path = f'{output_file}.txt'
    with open(file_path, 'w') as file:
        file.write(df_string)
        file.write('\n\n')
        file.write(f'Temporal Kappa={tmp_iaa_result.iaa}\n')
        file.write(f'Causal Kappa={cause_iaa_result.iaa}\n')
        file.write(f'Coref Kappa={coref_iaa_result.iaa}\n')
        file.write('\n\n')
        file.write('Most tmp unagreed mentions:\n')
        for mention, count in dict(sorted(tmp_iaa_result.unagreed.items(), key=lambda item: item[1], reverse=True)).items():
            file.write(f'{mention})={count}\n')

        file.write('\n\n')
        file.write(f'Num of mentions {annotator1}={len(annot_obj1.mentions)}, Num of mentions {annotator2}={len(annot_obj2.mentions)}\n')
        file.write(f'Num of pairs {annotator1}={annot_obj1.num_pairs} (Expected={annot_obj1.expected_pairs}), '
                   f'Num of pairs {annotator2}={annot_obj2.num_pairs} (Expected={annot_obj2.expected_pairs})\n')

    # print(f"DataFrame written to {file_path}")


def calculate_iaa(ments1, pairs1, ments2, pairs2):
    tmp_annot1, coref_annot1, cause_annot1 = get_annotations(pairs1)
    tmp_annot2, coref_annot2, cause_annot2 = get_annotations(pairs2)

    tmp_annot1_unpacked = [item[2] for item in tmp_annot1]
    tmp_annot2_unpacked = [item[2] for item in tmp_annot2]
    coref_annot1_unpacked = [item[2] for item in coref_annot1]
    coref_annot2_unpacked = [item[2] for item in coref_annot2]
    cause_annot1_unpacked = [item[2] for item in cause_annot1]
    cause_annot2_unpacked = [item[2] for item in cause_annot2]

    tmp_score = cohen_kappa_score(tmp_annot1_unpacked, tmp_annot2_unpacked)
    # print(f'Temporal Kappa={tmp_score}')
    tmp_diff, tmp_sames, tmp_unagreed = find_diffs(ments1, tmp_annot1, tmp_annot2)

    coref_score = 0 # cohen_kappa_score(coref_annot1_unpacked, coref_annot2_unpacked)
    coref_diff, coref_sames, coref_unagreed = [], [], 0 # find_diffs(ments1, coref_annot1, coref_annot2)
    # print(f'Temporal Kappa={coref_score}')

    cause_score = 0 # cohen_kappa_score(cause_annot1_unpacked, cause_annot2_unpacked)
    cause_diff, cause_sames, cause_unagreed = [], [], 0 # find_diffs(ments1, cause_annot1, cause_annot2)
    # print(f'Causal Score={cause_score}')

    tmp_iaa_result = IAAResultObj(tmp_score, tmp_diff, tmp_sames, tmp_unagreed)
    coref_iaa_result = IAAResultObj(coref_score, coref_diff, coref_sames, coref_unagreed)
    cause_iaa_result = IAAResultObj(cause_score, cause_diff, cause_sames, cause_unagreed)

    return tmp_iaa_result, coref_iaa_result, cause_iaa_result


def main(annot1_file, annot2_file):
    with open(annot1_file) as f:
        data1 = json.load(f)

    with open(annot2_file) as f:
        data2 = json.load(f)

    mentions1, num_pairs1, expected_pairs1 = count_stats_in_file(data1)
    annot_obj1 = AnnotObj(data1, mentions1, num_pairs1, expected_pairs1)

    mentions2, num_pairs2, expected_pairs2 = count_stats_in_file(data2)
    annot_obj2 = AnnotObj(data2, mentions2, num_pairs2, expected_pairs2)

    tmp_iaa_result, coref_iaa_result, cause_iaa_result = calculate_iaa(mentions1, data1["allPairs"], mentions2, data2["allPairs"])
    return annot_obj1, annot_obj2, tmp_iaa_result, coref_iaa_result, cause_iaa_result


if __name__ == "__main__":
    annotator1 = 'netta'
    annotator2 = 'michael'
    output_file = f'data/my_data/output/162d7_tmp_{annotator1}_{annotator2}_v1'
    _annot1_file = f'data/my_data/groups/group_b/162d7_temp_netta.json'
    _annot2_file = f'data/my_data/groups/group_b/162d7_temp_michael.json'

    _annot_obj1, _annot_obj2, _tmp_iaa_result, _coref_iaa_result, _cause_iaa_result = main(_annot1_file, _annot2_file)
    create_report(_annot_obj1, _annot_obj2, _tmp_iaa_result, _coref_iaa_result, _cause_iaa_result)
    print(f"Output written to {output_file}.txt")
