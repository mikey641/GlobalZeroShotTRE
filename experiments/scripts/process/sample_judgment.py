import json
import random

from scripts.process.calc_all_states import run_all_groups


def sample_for_validation(all_merged, sample_agreed_out, sample_test_out):
    sample = list(random.sample(all_merged, 50))
    sample.sort(key=lambda x: x[0])
    sample_test = list()
    for smp in sample:
        sample_test.append((smp[0], smp[1][0], smp[1][1], "my_decision="))

    with open(sample_agreed_out, 'w') as f:
        json.dump(sample, f, indent=4)

    with open(sample_test_out, 'w') as f:
        json.dump(sample_test, f, indent=4)


def main():
    sample_agreed_out = 'data/my_data/eval/sample_agreed.json'
    sample_test_out = 'data/my_data/eval/sample_test_annotated.json'
    all_merged = run_all_groups()
    sample_for_validation(all_merged, sample_agreed_out, sample_test_out)


if __name__ == "__main__":
    main()