import json
import string

from scripts.eval.model.evaluation import evaluation
from scripts.utils.classes.datasets_type import TBDDataset, EventFullDataset

if __name__ == "__main__":
    _prediction_file = "data/my_data/zero_shot/new_expr/eventfull_gemini-2.0-flash_4rels_cot_prompts_predictions.json"
    _data_type = EventFullDataset()

    _dataset_name = _data_type.get_name()
    _labels = _data_type.get_label_set()

    with open(_prediction_file) as _file:
        _data = json.load(_file)

    _pred_for_trans = {}
    _gold_for_trans = {}
    _all_golds = []
    _all_preds = []
    _count_nas = 0
    for key, value in _data.items():
        key_spt = key.split('#')
        doc_id = key_spt[0]
        source = key_spt[1]
        target = key_spt[2]
        gold = 'VAGUE' if value['gold_label'].upper() == 'UNCERTAIN' else value['gold_label'].upper()
        pred = value['target'].rstrip(string.whitespace + string.punctuation).upper()
        if pred == 'GENERATION FAILED':
            pred = 'BEFORE'
            _count_nas += 1

        _all_golds.append(_labels[gold])
        _all_preds.append(_labels[pred])

        if doc_id not in _pred_for_trans:
            _pred_for_trans[doc_id] = []
            _gold_for_trans[doc_id] = []

        _pred_for_trans[doc_id].append((source, _labels[pred], target))
        _gold_for_trans[doc_id].append((source, _labels[gold], target))

    evaluation(_all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _data_type)
    print(f"Number of Gold Relations: {len(_all_golds)}")
    print(f"Number of Predicted Relations: {len(_all_preds)}")
    print(f"Number of NAs: {_count_nas}")
    print("Done!")
