import json
from typing import final

import numpy as np

from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import NarrativeDataset, MATRES_DATASET_NAME, EVENTFULL_DATASET_NAME, \
    NARRATIVE_4RELS_DATASET_NAME, EventFullDataset
from scripts.utils.io_utils import read_file, read_pred_dot_file


def get_reverse_list(labels, pred_norm):
    n_classes = labels.get_num_classes()
    if n_classes == 4:
        return list((pred_norm[1], pred_norm[0], pred_norm[2], pred_norm[3]))
    elif n_classes == 6:
        return list((pred_norm[1], pred_norm[0], pred_norm[3], pred_norm[2], pred_norm[4], pred_norm[5]))
    else:
        raise ValueError('Labels length not supported!')


if __name__ == "__main__":
    _prediction_files = [
        "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_0.json",
        "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_1.json",
        "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_2.json",
        "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_3.json",
        "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_4.json",
    ]

    _dataset_type = EventFullDataset()

    _output_np_file = 'llms/voting/delete.npy'
    _output_json_file = 'llms/voting/delete.json'
    _test_docs_dict, _orig_ins_list = read_file(_dataset_type.get_test_file())

    _label_set = _dataset_type.get_label_set()

    gold_rels = dict()
    for inst in _orig_ins_list:
        doc_id = inst.docid.removesuffix('.json') if _dataset_type.get_name() == MATRES_DATASET_NAME else inst.docid
        source = inst.source.removeprefix('E')
        target = inst.target.removeprefix('E')
        label = inst.label
        gold_rels[f'{doc_id}#{source}#{target}'] = label

    aggregate_pred_as_dict = dict()
    for file_ in _prediction_files:
        _pred_as_dict, _ = read_pred_dot_file(file_, _test_docs_dict, _dataset_type)
        for key, value in _pred_as_dict.items():
            if key not in aggregate_pred_as_dict:
                if _dataset_type.get_name() in [EVENTFULL_DATASET_NAME, MATRES_DATASET_NAME, NARRATIVE_4RELS_DATASET_NAME]:
                    aggregate_pred_as_dict[key] = [0.0,0.0,0.0,0.0]
                else:
                    aggregate_pred_as_dict[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            aggregate_pred_as_dict[key][value] += 1

    final_golds = []
    final_preds = []

    # assert len(gold_rels) == len(aggregate_pred_as_dict)
    for ins in _orig_ins_list:
        doc_id = ins.docid
        source = ins.source
        target = ins.target
        label = ins.label
        sentdiff = ins.sentdiff
        key = f'{doc_id}#{source}#{target}'
        rev_key = f'{doc_id}#{target}#{source}'
        if sentdiff > 1:
            if key in aggregate_pred_as_dict:
                final_golds.append(_dataset_type.get_label_set()[label])
                final_preds.append(np.argmax(aggregate_pred_as_dict[key]))
            elif rev_key in aggregate_pred_as_dict:
                final_golds.append(_dataset_type.get_label_set()[_dataset_type.get_label_set().get_reverse_label(label)])
                final_preds.append(np.argmax(aggregate_pred_as_dict[rev_key]))
            else:
                raise KeyError(f'Key {key} not found in aggregate predictions and reverse key {rev_key} not found in aggregate predictions!')

    evaluation(final_golds, final_preds, None, None, _dataset_type)

    print('Done!')
