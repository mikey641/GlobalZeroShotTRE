import json
import random

import pandas as pd

from scripts.eval.calc_all_states import run_all_groups, groups


def extract_file_agree_pairs(group_a_tmp_results):
    merged = list()
    for file_name, result_tup in group_a_tmp_results.items():
        iaa_same = result_tup[2].same
        iaa_same_with_file = [(file_name, iaa) for iaa in iaa_same]
        merged.extend(iaa_same_with_file)

    return merged


def sample_for_validation(groups_files, sample_agreed_out, sample_test_out):
    all_merged = list()
    for group in groups:
        all_merged.extend(extract_file_agree_pairs(groups_files[group][1]))

    sample = list(random.sample(all_merged, 50))
    sample.sort(key=lambda x: x[0])
    sample_test = list()
    for smp in sample:
        sample_test.append((smp[0], smp[1][0], smp[1][1], "my_decision="))

    with open(sample_agreed_out, 'w') as f:
        json.dump(sample, f, indent=4)

    with open(sample_test_out, 'w') as f:
        json.dump(sample_test, f, indent=4)


def generate_files_for_judge(groups_files, judge_folder):
    for group in groups:
        group_objects = groups_files[group][1]
        for topic_name, topic_obj in group_objects.items():
            diffs_list = list()
            diff_pairs = topic_obj[2].diff
            for pair in diff_pairs:
                diffs_list.append([pair[0], pair[1], ""])

            df = pd.DataFrame(columns=['Mention1', 'Mention2', 'Started First?'], data=diffs_list)
            df.to_csv(f'{judge_folder}/{group}/{topic_name}.csv', sep=',', index=False, encoding='utf-8')
            # with open(f'{judge_folder}/{group}/{topic_name}.txt', 'w') as f:
            #     f.write(tabulate(df, showindex=False, headers=df.columns, tablefmt='grid'))
            #     f.write('\n')


def main(run_valid, run_judge):
    sample_agreed_out = 'data/my_data/eval/sample_agreed.json'
    sample_test_out = 'data/my_data/eval/sample_test_annotated.json'
    judge_folder = 'data/my_data/judge/'

    groups_files = run_all_groups()
    if run_valid:
        sample_for_validation(groups_files, sample_agreed_out, sample_test_out)

    if run_judge:
        generate_files_for_judge(groups_files, judge_folder)


if __name__ == "__main__":
    _run_valid = False
    _run_judge = True
    main(_run_valid, _run_judge)
