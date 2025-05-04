import json

import numpy as np

from scripts.eval.prompt.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import TBDDataset, MATRES_DATASET_NAME, NarrativeDataset, MatresDataset, \
    EventFullDataset
from scripts.utils.io_utils import read_file


def run_process(gold_as_dict, gold_order_list, pred_order_list, predictions, dataset_type):
    labels = dataset_type.get_label_set()

    pred_as_dict = dict()
    for idx, pred in enumerate(pred_order_list):
        key = f'{pred[0]}#{pred[1]}#{pred[3]}'
        pred_as_dict[key] = predictions[idx]

    _, _, gold_for_trans, pred_for_trans, _count_nas = convert_format(gold_order_list, pred_as_dict, labels, debug=False)

    all_golds = []
    all_preds = []
    for key, value in gold_as_dict.items():
        all_golds.append(gold_as_dict[key])
        all_preds.append(pred_as_dict[key])

    evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type)


if __name__ == "__main__":
    # Load numpy file
    _data_type = MatresDataset()
    _predictions = np.load('data/my_data/predictions/gen_allNew_best_roberta_matres_100_100_0.1.npy')

    # load order list json file
    # gold_order will be used to evaluate the predictions, pred_order will be used to generate the predictions
    with open('data/my_data/predictions/gen_allNew_order_list_best_roberta_matres_100_100_0.1.json', 'r') as f:
        _pred_order_list = json.load(f)

    _test_docs_dict, _gold_order_list = read_file(_data_type.get_test_file())

    _gold_as_dict = {}
    for idx, item in enumerate(_gold_order_list):
        docid = item.docid.removesuffix('.json') if _data_type.get_name() == MATRES_DATASET_NAME else item.docid
        source = item.source.removeprefix('E') if _data_type.get_name() == MATRES_DATASET_NAME else item.source
        target = item.target.removeprefix('E') if _data_type.get_name() == MATRES_DATASET_NAME else item.target

        # key: doc_id#source_event#target_event, value: relation
        _gold_as_dict[f'{docid}#{source}#{target}'] = _data_type.get_label_set()[item.label]

    for item in _pred_order_list:
        item[0] = item[0].removesuffix('.json') if _data_type.get_name() == MATRES_DATASET_NAME else item[0]

    run_process(_gold_as_dict, _gold_order_list, _pred_order_list, _predictions, _data_type)
