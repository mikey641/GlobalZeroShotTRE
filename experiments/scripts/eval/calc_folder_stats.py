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
    group_files = gather_files('data/my_data/files_count_annot')
    total_files = 0
    _total_ment, _total_pairs, _total_diff, _total_same, _avg_agreement, _total_diff_same = 0, 0, 0, 0, 0, 0

    total_tmp_pairs = sum([iaa_obj[1] for iaa_obj in group_files.values()])
    avg_tmp_pairs = total_tmp_pairs / len(group_files)

    total_coref_pairs = sum([iaa_obj[2] for iaa_obj in group_files.values()])
    avg_coref_pairs = total_coref_pairs / len(group_files)

    total_cause_pairs = sum([iaa_obj[3] for iaa_obj in group_files.values()])
    avg_cause_pairs = total_cause_pairs / len(group_files)

    print(f'Total-Files: {len(group_files)}')
    print(f'Total-Temp-pairs: {total_tmp_pairs}')
    print(f'Avg-Temp-pairs: {avg_tmp_pairs}')
    print(f'Total-Coref-pairs: {total_coref_pairs}')
    print(f'Avg-Coref-pairs: {avg_coref_pairs}')
    print(f'Total-Cause-pairs: {total_cause_pairs}')
    print(f'Avg-Cause-pairs: {avg_cause_pairs}')

    print('---------------------------------')


if __name__ == "__main__":
    main()
