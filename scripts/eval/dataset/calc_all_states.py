import os

from scripts.eval.dataset import calc_iaa

# groups = ['group_a', 'group_b', 'group_c']
# groups = ['g_demo_a', 'g_demo_b', 'g_demo_c']
# PATH = f'data/my_data/groups_exports/'
groups = ['cc_a', 'cc_b', 'cc_c']
PATH = f'data/my_data/demo_files/groups/'


def gather_files(path):
    group_files = dict()
    for file in os.listdir(path):
        with open(f'{path}/{file}', 'r') as f:
            if file.startswith('_'):
                continue

            file_group_name = file.split('_')[0]
            if file_group_name not in group_files:
                group_files[file_group_name] = list()
            group_files[file_group_name].append(f'{path}/{file}')

    return group_files


def read_group_files(group):
    # path = f'data/my_data/groups_exports/{group}/'
    group_files = gather_files(f'{PATH}/{group}/')
    group_tmp_results = dict()
    for file_group_name, files in group_files.items():
        if len(files) != 2:
            raise ValueError(f'Expected 2 files for group {file_group_name}, but found {len(files)}')

        annot_obj1, annot_obj2, tmp_iaa_result, coref_iaa_result, cause_iaa_result = calc_iaa.main(files[0], files[1])
        group_tmp_results[file_group_name] = (annot_obj1, annot_obj2, tmp_iaa_result, coref_iaa_result, cause_iaa_result)
    return group_files, group_tmp_results


def print_group_stats(group, group_a_files, group_a_tmp_results):
    total_ment = sum([len(iaa_obj[0].mentions) for iaa_obj in group_a_tmp_results.values()])
    total_tmp_pairs = sum([iaa_obj[0].num_tmp_pairs for iaa_obj in group_a_tmp_results.values()])
    total_diff = sum([len(iaa_obj[2].diff) for iaa_obj in group_a_tmp_results.values()])
    total_same = sum([len(iaa_obj[2].same) for iaa_obj in group_a_tmp_results.values()])
    avg_agreement_tmp = sum([iaa_obj[2].iaa for iaa_obj in group_a_tmp_results.values()]) / len(group_a_tmp_results)
    avg_agreement_causal = sum([iaa_obj[4].iaa for iaa_obj in group_a_tmp_results.values()]) / len(group_a_tmp_results)

    total_diff_same = total_diff + total_same

    print(group)
    print(f'Total-Files: {len(group_a_files)}')
    print(f'Total-mentions: {total_ment}')
    print(f'Total-Temp-pairs: {total_tmp_pairs}')
    print(f'Avg-Temp-pairs: {total_tmp_pairs/len(group_a_files)}')
    print(f'Total-disagree: {total_diff}')
    print(f'Total-agree: {total_same}')
    print(f'Total-(diff + same): {total_diff_same}')
    print(f'Average tmp agreement: {avg_agreement_tmp}')
    print(f'Average causal agreement: {avg_agreement_causal}')
    print('---------------------------------')
    return total_ment, total_tmp_pairs, total_diff, total_same, avg_agreement_tmp, total_diff_same


def run_all_groups():
    total_files = 0
    total_ment, total_pairs, total_diff, total_same, avg_agreement, total_diff_same = 0, 0, 0, 0, 0, 0
    groups_files = dict()
    for group in groups:
        groups_files[group] = read_group_files(group)

    for group in groups:
        _total_ment, _total_pairs, _total_diff, _total_same, _avg_agreement, _total_diff_same = print_group_stats(group, groups_files[group][0], groups_files[group][1])
        total_files += len(groups_files[group][0])
        total_ment += _total_ment
        total_pairs += _total_pairs
        total_diff += _total_diff
        total_same += _total_same
        avg_agreement += _avg_agreement
        total_diff_same += _total_diff_same

    print('Total')
    print(f'Total-Files: {total_files}')
    print(f'Total-mentions: {total_ment}')
    print(f'avg-mentions: {total_ment/total_files}')
    print(f'Total-pairs: {total_pairs}')
    print(f'Total-disagree: {total_diff}')
    print(f'Total-agree: {total_same}')
    print(f'Total-(diff + same): {total_diff_same}')
    print(f'Average agreement: {avg_agreement / len(groups)}')
    print('---------------------------------')

    return groups_files


if __name__ == "__main__":
    run_all_groups()
