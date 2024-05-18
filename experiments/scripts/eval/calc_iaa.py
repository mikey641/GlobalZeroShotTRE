# load json file
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.eval.utils import count_stats_in_file, get_annotations, find_diffs


def create_report(data1, mentions1, num_pairs1, expected_pairs1, data2, mentions2, num_pairs2, expected_pairs2,
                  tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed):
    df = pd.DataFrame(columns=['Mention1', 'Mention2', annotator1, annotator2], data=tmp_diff)
    df_string = df.to_string(index=False)
    file_path = f'{output_file}.txt'
    with open(file_path, 'w') as file:
        file.write(df_string)
        file.write('\n\n')
        file.write(f'Temporal Kappa={tmp_score}\n')
        file.write(f'Causal Kappa={cause_score}\n')
        file.write(f'Coref Kappa={coref_score}\n')
        file.write('\n\n')
        file.write('Most tmp unagreed mentions:\n')
        for mention, count in dict(sorted(tmp_unagreed.items(), key=lambda item: item[1], reverse=True)).items():
            file.write(f'{mention})={count}\n')

        file.write('\n\n')
        file.write(f'Num of mentions {annotator1}={len(mentions1)}, Num of mentions {annotator2}={len(mentions2)}\n')
        file.write(f'Num of pairs {annotator1}={num_pairs1} (Expected={expected_pairs1}), Num of pairs {annotator2}={num_pairs2} (Expected={expected_pairs2})\n')

    print(f"DataFrame written to {file_path}")


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
    print(f'Temporal Kappa={tmp_score}')
    tmp_diff, tmp_unagreed = find_diffs(ments1, tmp_annot1, tmp_annot2)

    coref_score = cohen_kappa_score(coref_annot1_unpacked, coref_annot2_unpacked)
    coref_diff, coref_unagreed = find_diffs(ments1, coref_annot1, coref_annot2)
    print(f'Temporal Kappa={coref_score}')

    cause_score = cohen_kappa_score(cause_annot1_unpacked, cause_annot2_unpacked)
    cause_diff, cause_unagreed = find_diffs(ments1, cause_annot1, cause_annot2)
    print(f'Causal Score={cause_score}')
    return tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed


def main():
    with open(annot1_file) as f:
        data1 = json.load(f)

    with open(annot2_file) as f:
        data2 = json.load(f)

    print('81d6_rel_FinalAnnotations_1')
    mentions1, num_pairs1, expected_pairs1 = count_stats_in_file(data1)
    print('81d6_rel_FinalAnnotations_1')
    mentions2, num_pairs2, expected_pairs2 = count_stats_in_file(data2)

    tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed = calculate_iaa(mentions1, data1["allPairs"], mentions2, data2["allPairs"])
    create_report(data1, mentions1, num_pairs1, expected_pairs1, data2, mentions2, num_pairs2, expected_pairs2,
                  tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed)


if __name__ == "__main__":
    annotator1 = 'michael'
    annotator2 = 'benji'
    output_file = f'data/my_data/output/68d7_tmp_{annotator1}_{annotator2}_v1'
    annot1_file = f'data/my_data/benji_michael/68d7_temp_michael.json'
    annot2_file = f'data/my_data/benji_michael/68d7_temp_benji.json'
    main()
