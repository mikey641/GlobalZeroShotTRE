import json
import os

from scripts.eval.utils import count_stats_in_file


def gather_files(path):
    group_files = dict()
    for file in os.listdir(path):
        with open(f'{path}/{file}', 'r') as f:
            data = json.load(f)
            group_files[file] = count_stats_in_file(data)

    return group_files


def main():
    group_files = gather_files('data/my_data/EventFullTrainExports')
    total_files = 0
    _total_ment, _total_pairs, _total_diff, _total_same, _avg_agreement, _total_diff_same = 0, 0, 0, 0, 0, 0

    total_mentions = sum([len(iaa_obj[0]) for iaa_obj in group_files.values()])
    avg_mentions = total_mentions / len(group_files)

    total_tmp_pairs = sum([iaa_obj[1] for iaa_obj in group_files.values()])
    avg_tmp_pairs = total_tmp_pairs / len(group_files)

    total_equal_pairs = sum([iaa_obj[2] for iaa_obj in group_files.values()])
    avg_equal_pairs = total_equal_pairs / len(group_files)

    total_before_after_pairs = sum([iaa_obj[3] for iaa_obj in group_files.values()])
    avg_before_after_pairs = total_before_after_pairs / len(group_files)

    total_vague_pairs = sum([iaa_obj[4] for iaa_obj in group_files.values()])
    avg_vague_pairs = total_vague_pairs / len(group_files)

    total_expected_pairs = sum([iaa_obj[5] for iaa_obj in group_files.values()])

    print(f'Total-Files: {len(group_files)}')
    print(f'Total-mentions: {total_mentions}')
    print(f'Avg-File-mentions: {avg_mentions}')
    print(f'Total-Temp-pairs: {total_tmp_pairs}')
    print(f'Avg-File-Temp-pairs: {avg_tmp_pairs}')
    print(f'Total-Expected-pairs: {total_expected_pairs}')
    print(f'Total-Equal-pairs: {total_equal_pairs}')
    print(f'Avg-File-Equal-pairs: {avg_equal_pairs}')
    print(f'Total-Before/After-pairs: {total_before_after_pairs}')
    print(f'Avg-File-Before/After-pairs: {avg_before_after_pairs}')
    print(f'Total-Uncertain-pairs: {total_vague_pairs}')
    print(f'Avg-File-Uncertain-pairs: {avg_vague_pairs}')

    print('---------------------------------')


if __name__ == "__main__":
    main()
