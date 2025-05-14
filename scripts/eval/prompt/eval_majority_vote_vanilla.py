import json
from typing import final

import numpy as np

from scripts.eval.prompt.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import EventFullDataset, MATRES_DATASET_NAME, EVENTFULL_DATASET_NAME, \
    NARRATIVE_4RELS_DATASET_NAME, NarrativeDataset, MatresDataset, TBDDataset
from scripts.utils.io_utils import read_file, read_pred_dot_file


def get_reverse_list(labels, pred_norm):
    n_classes = labels.get_num_classes()
    if n_classes == 4:
        return list((pred_norm[1], pred_norm[0], pred_norm[2], pred_norm[3]))
    elif n_classes == 6:
        return list((pred_norm[1], pred_norm[0], pred_norm[3], pred_norm[2], pred_norm[4], pred_norm[5]))
    else:
        raise ValueError('Labels length not supported!')


def eval_full_doc(orig_ins_list, gold_rels, aggregate_pred_as_dict, dataset_type):
    final_golds = []
    final_preds = []
    pred_as_dict = dict()
    for key in aggregate_pred_as_dict.keys():
        pred_as_dict[key] = np.argmax(aggregate_pred_as_dict[key])
        if key in gold_rels:
            key_spl = key.split("#")
            gold_rel_val = dataset_type.get_label_set()[gold_rels[key]]
            final_golds.append(gold_rel_val)
            final_preds.append(pred_as_dict[key])
    _, _, gold_for_trans, pred_for_trans, _ = convert_format(orig_ins_list, pred_as_dict, dataset_type.get_label_set(), debug=False)
    return evaluation(final_golds, final_preds, gold_for_trans, pred_for_trans, dataset_type)


def eval_sentdiff(orig_ins_list, aggregate_pred_as_dict, dataset_type, consecutive):
    final_golds = []
    final_preds = []
    for ins in orig_ins_list:
        doc_id = ins.docid
        source = ins.source
        target = ins.target
        label = ins.label
        sentdiff = ins.sentdiff
        key = f'{doc_id}#{source}#{target}'
        rev_key = f'{doc_id}#{target}#{source}'

        condition = False
        if consecutive and sentdiff <= 1:
            condition = True
        elif not consecutive and sentdiff > 1:
            condition = True

        if condition:
            if key in aggregate_pred_as_dict:
                final_golds.append(dataset_type.get_label_set()[label])
                final_preds.append(np.argmax(aggregate_pred_as_dict[key]))
            elif rev_key in aggregate_pred_as_dict:
                final_golds.append(dataset_type.get_label_set()[dataset_type.get_label_set().get_reverse_label(label)])
                final_preds.append(np.argmax(aggregate_pred_as_dict[rev_key]))
            else:
                raise KeyError(f'Key {key} not found in aggregate predictions and reverse key {rev_key} not found in aggregate predictions!')

    return evaluation(final_golds, final_preds, None, None, dataset_type)



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

    _gold_rels = dict()
    for inst in _orig_ins_list:
        doc_id = inst.docid.removesuffix('.json') if _dataset_type.get_name() == MATRES_DATASET_NAME else inst.docid
        source = inst.source.removeprefix('E')
        target = inst.target.removeprefix('E')
        label = inst.label
        _gold_rels[f'{doc_id}#{source}#{target}'] = label

    _aggregate_pred_as_dict = dict()
    for file_ in _prediction_files:
        _pred_as_dict, _ = read_pred_dot_file(file_, _test_docs_dict, _dataset_type)
        for key, value in _pred_as_dict.items():
            if key not in _aggregate_pred_as_dict:
                if _dataset_type.get_name() in [EVENTFULL_DATASET_NAME, MATRES_DATASET_NAME, NARRATIVE_4RELS_DATASET_NAME]:
                    _aggregate_pred_as_dict[key] = [0.0,0.0,0.0,0.0]
                else:
                    _aggregate_pred_as_dict[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            _aggregate_pred_as_dict[key][value] += 1

    print('\n\n###### Evaluation Results Full Document ########')
    f1_full = eval_full_doc(_orig_ins_list, _gold_rels, _aggregate_pred_as_dict, _dataset_type)

    print('\n\n###### Evaluation Results Consecutive Sentences ########')
    f1_consec = eval_sentdiff(_orig_ins_list, _aggregate_pred_as_dict, _dataset_type, consecutive=True)
    print('\n\n###### Evaluation Results Non-Consecutive Sentences ########')
    f1_non_consec = eval_sentdiff(_orig_ins_list, _aggregate_pred_as_dict, _dataset_type, consecutive=False)

    print('\n\n###### Summary ######')
    print(f"Full F1: {f1_full}")
    print(f"Consecutive F1: {f1_consec}")
    print(f"Non-Consecutive F1: {f1_non_consec}")
    print('Done!')
