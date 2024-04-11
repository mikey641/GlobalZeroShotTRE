# load json file
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from scripts.eval.utils import count_stats_in_file, get_annotations, find_diffs


def create_report(data1, num_mentions1, num_pairs1, expected_pairs1, data2, num_mentions2, num_pairs2, expected_pairs2,
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
        file.write(f'Num of mentions {annotator1}={num_mentions1}, Num of mentions {annotator2}={num_mentions2}\n')
        file.write(f'Num of pairs {annotator1}={num_pairs1} (Expected={expected_pairs1}), Num of pairs {annotator2}={num_pairs2} (Expected={expected_pairs2})\n')

    print(f"DataFrame written to {file_path}")


def calculate_iaa(data1, data2):
    tmp_annot1, coref_annot1, cause_annot1 = get_annotations(data1)
    tmp_annot2, coref_annot2, cause_annot2 = get_annotations(data2)

    tmp_score = cohen_kappa_score(tmp_annot1, tmp_annot2)
    print(f'Temporal Kappa={tmp_score}')
    tmp_diff, tmp_unagreed = find_diffs(data1, tmp_annot1, tmp_annot2)

    coref_score = cohen_kappa_score(coref_annot1, coref_annot2)
    coref_diff, coref_unagreed = find_diffs(data1, coref_annot1, coref_annot2)
    print(f'Temporal Kappa={coref_score}')

    cause_score = cohen_kappa_score(cause_annot1, cause_annot2)
    cause_diff, cause_unagreed = find_diffs(data1, cause_annot1, cause_annot2)
    print(f'Causal Score={cause_score}')
    return tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed


def main():
    with open(annot1_file) as f:
        data1 = json.load(f)

    with open(annot2_file) as f:
        data2 = json.load(f)

    print('81d6_rel_FinalAnnotations_1')
    num_mentions1, num_pairs1, expected_pairs1 = count_stats_in_file(data1)
    print('81d6_rel_FinalAnnotations_1')
    num_mentions2, num_pairs2, expected_pairs2 = count_stats_in_file(data2)

    tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed = calculate_iaa(data1, data2)
    create_report(data1, num_mentions1, num_pairs1, expected_pairs1, data2, num_mentions2, num_pairs2, expected_pairs2,
                  tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed)


if __name__ == "__main__":
    annotator1 = 'netta'
    annotator2 = 'michael'
    output_file = f'data/my_data/output/148d5_tmp_{annotator1}_{annotator2}_v2'
    annot1_file = f'data/my_data/input/148d5_temp_netta.json'
    annot2_file = f'data/my_data/input/148d5_temp_michael.json'
    main()
