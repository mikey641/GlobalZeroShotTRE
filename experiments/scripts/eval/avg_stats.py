import json
import os

from scripts.eval.calc_iaa import calculate_iaa
from scripts.eval.utils import count_stats_in_file


def main():
    tot_mentions = list()
    tot_pairs = list()
    tot_expected_pairs = list()
    temp_kappas = list()
    temporal_annotations = list()
    compare_count = 0
    done = set()
    for file1 in os.listdir(input_folder):
        for file2 in os.listdir(input_folder):
            if file1 == file2:
                continue

            split_file1 = file1.split("_")
            split_file2 = file2.split("_")
            if split_file1[0] == split_file2[0] and split_file1[0] not in done:
                with open(input_folder + "/" + file1) as f:
                    data1 = json.load(f)

                with open(input_folder + "/" + file2) as f:
                    data2 = json.load(f)

                annot1 = split_file1[-1].split(".")[0]
                annot2 = split_file2[-1].split(".")[0]
                print(f'{annot1} {split_file1[0]}_rel_FinalAnnotations_1')
                num_mentions1, num_pairs1, expected_pairs1 = count_stats_in_file(data1)
                print(f'{annot2} {split_file2[0]}_rel_FinalAnnotations_2')
                num_mentions2, num_pairs2, expected_pairs2 = count_stats_in_file(data2)

                if num_mentions1 != num_mentions2 or num_pairs1 != num_pairs2 or expected_pairs1 != expected_pairs2:
                    print(f'Annotations are not equal for {annot1} and {annot2}')
                    return

                tmp_score, tmp_diff, tmp_unagreed, coref_score, coref_diff, coref_unagreed, cause_score, cause_diff, cause_unagreed = calculate_iaa(
                    data1, data2)

                if split_file1[0] not in dont_consider_for_tmp_annotations:
                    pair_temp_made_avg = (data1['_tempAnnotationMade'] + data2['_tempAnnotationMade']) / 2
                    temporal_annotations.append(pair_temp_made_avg)

                temp_kappas.append(tmp_score)
                tot_mentions.append(num_mentions1)
                tot_pairs.append(num_pairs1)
                tot_expected_pairs.append(expected_pairs1)
                compare_count += 1
                done.add(split_file1[0])
                print(f'{annot1}/{annot2} {split_file1[0]} temp kappa={tmp_score}')

    print()
    print(f'Number of pairs of annotated files={compare_count}')
    print(f'Total mentions={sum(tot_mentions)}')
    print(f'Average mentions={sum(tot_mentions) / len(tot_mentions)}')
    print(f'Total pairs={sum(tot_pairs)}')
    print(f'Average pairs per file={sum(tot_pairs) / len(tot_pairs)}')
    print(f'Total expected pairs={sum(tot_expected_pairs)}')
    print(f'Average Temporal Kappa={sum(temp_kappas) / len(temp_kappas)}')
    print(f'Average Temporal Annotation Made={sum(temporal_annotations) / len(temporal_annotations)}')


if __name__ == "__main__":
    dont_consider_for_tmp_annotations = ['7d7', '17d7', '134d8', '135d4']
    annotator1 = 'netta'
    annotator2 = 'michael'
    output_file = f'data/my_data/output/148d5_tmp_{annotator1}_{annotator2}_v2'
    input_folder = f'data/my_data/michael_netta'
    main()
