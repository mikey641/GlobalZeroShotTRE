import numpy as np

from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, OMNITEMP_DATASET_NAME, OmniTempDataset
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
    _prediction_files = [
        "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_1.json",
        "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_2.json",
        "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_3.json",
        "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_4.json",
        "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_5.json",
    ]

    _dataset_type = OmniTempDataset()

    _output_np_file = 'llms/voting/delete.npy'
    _output_json_file = 'llms/voting/delete.json'
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
