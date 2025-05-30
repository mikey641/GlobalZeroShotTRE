import json
import string

from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import TBDDataset, EventFullDataset, NarrativeDataset, MatresDataset
from scripts.utils.io_utils import load_json_lines, read_file


def from_jsonl_to_dict(loaded_data):
    """
    Convert loaded data from jsonl to dict
    """
    data = dict()
    for line in loaded_data:
        data[line['key']] = {"target": line['target'], "gold_label": line['gold_label']}

    return data


def prepare_dicts(key, value, all_golds, all_preds, pred_for_trans, gold_for_trans, labels):
    count_nas = 0
    gold = 'VAGUE' if value['gold_label'].upper() == 'UNCERTAIN' else value['gold_label'].upper()
    pred = value['target'].rstrip(string.whitespace + string.punctuation).upper()
    if pred == 'GENERATION FAILED':
        pred = 'BEFORE'
        count_nas = 1
    all_golds.append(labels[gold])
    all_preds.append(labels[pred])

    key_split = key.split("#")
    if key_split[0] not in pred_for_trans:
        pred_for_trans[key_split[0]] = []
        gold_for_trans[key_split[0]] = []

    pred_for_trans[key_split[0]].append((key_split[1], labels[pred], key_split[2]))
    gold_for_trans[key_split[0]].append((key_split[1], labels[gold], key_split[2]))
    return count_nas


def run_eval_all(data, data_type):
    pred_for_trans = {}
    gold_for_trans = {}
    all_golds = []
    all_preds = []
    labels = data_type.get_label_set()
    count_nas = 0
    for key, value in data.items():
        count_nas = prepare_dicts(key, value, all_golds, all_preds, pred_for_trans, gold_for_trans, labels)
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, data_type)

    print(f"Number of Gold Relations: {len(all_golds)}")
    print(f"Number of Predicted Relations: {len(all_preds)}")
    print(f"Number of NAs: {count_nas}")
    return f1


def run_eval_sentdiff(data_type, gold_dict_send_diff, consecutive):
    pred_for_trans = {}
    gold_for_trans = {}
    all_golds = []
    all_preds = []
    labels = data_type.get_label_set()
    count_nas = 0
    for key, value in _data.items():
        sentdiff = gold_dict_send_diff[key]
        if (consecutive and sentdiff <= 1) or (not consecutive and sentdiff > 1):
            count_nas += prepare_dicts(key, value, all_golds, all_preds, pred_for_trans, gold_for_trans, labels)
    f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, data_type)

    print(f"Number of Gold Relations: {len(all_golds)}")
    print(f"Number of Predicted Relations: {len(all_preds)}")
    print(f"Number of NAs: {count_nas}")
    return f1


if __name__ == "__main__":
    _prediction_file = "data/my_data/zero_shot/new_expr/omni/omni_Meta-Llama-3-8B-Instruct_run_CoT_predictions.jsonl"
    _data_type = EventFullDataset()

    if _prediction_file.endswith(".jsonl"):
        json_lines = load_json_lines(_prediction_file)
        _data = from_jsonl_to_dict(json_lines)
    else:
        with open(_prediction_file) as _file:
            _data = json.load(_file)

    _dataset_name = _data_type.get_name()
    _labels = _data_type.get_label_set()
    _test_docs_dict, _orig_ins_list = read_file(_data_type.get_test_file())

    _gold_dict_send_diff = {}
    for ins in _orig_ins_list:
        _doc_id = ins.docid
        _source = ins.source.removeprefix("E")
        _target = ins.target.removeprefix("E")
        _sentdiff = ins.sentdiff
        _key = f'{_doc_id}#{_source}#{_target}'
        _rev_key = f'{_doc_id}#{_target}#{_source}'
        if _key not in _gold_dict_send_diff:
            _gold_dict_send_diff[_key] = _sentdiff
        if _rev_key not in _gold_dict_send_diff:
            _gold_dict_send_diff[_rev_key] = _sentdiff

    print('\n\n###### Running eval for all ######')
    f1_full = run_eval_all(_data, _data_type)
    print('\n\n###### Running eval for Consecutive sentences ######')
    f1_consec = run_eval_sentdiff(_data_type, _gold_dict_send_diff, consecutive=True)
    print('\n\n###### Running eval for Non-consecutive sentences ######')
    f1_non_consec = run_eval_sentdiff(_data_type, _gold_dict_send_diff, consecutive=False)

    print('\n\n###### Summary ######')
    print(f"Full F1: {f1_full}")
    print(f"Consecutive Sentences F1: {f1_consec}")
    print(f"Non-Consecutive Sentences F1: {f1_non_consec}")

    print("Done!")
