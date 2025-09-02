import argparse
import os

import numpy as np

from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, OMNITEMP_DATASET_NAME, OmniTempDataset, \
    NarrativeDataset, MatresDataset, TBDDataset, MavenDataset
from scripts.utils.io_utils import read_file, read_pred_dot_file, load_golds


def eval_full_doc(test_as_dict, aggregate_pred_as_dict, dataset_type):
    final_golds = []
    final_preds = []
    pred_as_dict = dict()
    for key in aggregate_pred_as_dict.keys():
        pred_as_dict[key] = np.argmax(aggregate_pred_as_dict[key])
        if key in test_as_dict:
            gold_rel_val = test_as_dict[key]
            final_golds.append(gold_rel_val)
            final_preds.append(pred_as_dict[key])
    _, _, gold_for_trans, pred_for_trans, _ = convert_format(test_as_dict, pred_as_dict, dataset_type.get_label_set(), debug=False)
    return evaluation(final_golds, final_preds, gold_for_trans, pred_for_trans, dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run llm on test data")
    parser.add_argument("--test_ds", help="The test database name (nt, matres, omni, tbd)", type=str, required=True)
    parser.add_argument("--test_folder", help="Test folder, should only contain the files to consider", type=str, required=True)

    args = parser.parse_args()
    print(vars(args))

    _test_folder = args.test_folder
    _test_ds = args.test_ds

    if args.test_ds == "nt":
        _dataset_type = NarrativeDataset()
    elif args.test_ds == "matres":
        _dataset_type = MatresDataset()
    elif args.test_ds == "omni":
        _dataset_type = OmniTempDataset()
    elif args.test_ds == "tbd":
        _dataset_type = TBDDataset()
    elif args.test_ds == "maven":
        _dataset_type = MavenDataset()
    else:
        raise ValueError("Invalid test database name.")

    _prediction_files = [f"{_test_folder}/{file_}" for file_ in os.listdir(args.test_folder) if file_.endswith('.json')]

    if len(_prediction_files) == 0:
        raise ValueError("No prediction files found in the specified folder.")

    _label_set = _dataset_type.get_label_set()
    _test_as_dict, _all_gold_files = load_golds(_dataset_type.get_test_file(), _label_set)

    _aggregate_pred_as_dict = dict()
    for file_ in _prediction_files:
        _pred_as_dict, _ = read_pred_dot_file(file_, _all_gold_files, _dataset_type)
        for key, value in _pred_as_dict.items():
            if key not in _aggregate_pred_as_dict:
                if _dataset_type.get_name() in [OMNITEMP_DATASET_NAME, MATRES_DATASET_NAME]:
                    _aggregate_pred_as_dict[key] = [0.0,0.0,0.0,0.0]
                else:
                    _aggregate_pred_as_dict[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            _aggregate_pred_as_dict[key][value] += 1

    print('\n\n###### Evaluation Results Full Document ########')
    f1_full = eval_full_doc(_test_as_dict, _aggregate_pred_as_dict, _dataset_type)

    print('\n\n###### Summary ######')
    print(f"Full F1: {f1_full}")
    print('Done!')
