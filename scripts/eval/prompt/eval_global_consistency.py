import time
from itertools import permutations

import numpy as np

from scripts.eval.prompt.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.eval.shared.gurobi_optimizer_entrop import run_transitive_constraints
from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, EVENTFULL_DATASET_NAME, \
    NARRATIVE_4RELS_DATASET_NAME, MatresDataset, EventFullDataset, NarrativeDataset, TBDDataset
from scripts.utils.io_utils import read_pred_dot_file, read_file


def get_reverse_list(labels, pred_norm):
    n_classes = labels.get_num_classes()
    if n_classes == 4:
        return list((pred_norm[1], pred_norm[0], pred_norm[2], pred_norm[3]))
    elif n_classes == 6:
        return list((pred_norm[1], pred_norm[0], pred_norm[3], pred_norm[2], pred_norm[4], pred_norm[5]))
    else:
        raise ValueError('Labels length not supported!')


def adjust_pred_and_rev_preds(aggregate_pred_as_dict, labels):
    handeled_keys = set()
    for key, value in aggregate_pred_as_dict.items():
        if key in handeled_keys:
            continue
        split = key.split("#")
        rev_key = f'{split[0]}#{split[2]}#{split[1]}'
        aggregate_pred_as_dict[key] = aggregate_pred_as_dict[key] + get_reverse_list(labels, aggregate_pred_as_dict[rev_key])
        aggregate_pred_as_dict[rev_key] = aggregate_pred_as_dict[rev_key] + get_reverse_list(labels, aggregate_pred_as_dict[key])
        handeled_keys.add(rev_key)


def run_majority_vote_trans_const(dataset_type, test_docs_dict, prediction_files, orig_ins_list):
    gold_rels = dict()
    event_dict = dict()
    for inst in orig_ins_list:
        doc_id = inst.docid if dataset_type.get_name() != MATRES_DATASET_NAME else inst.docid.removesuffix(".json")
        source = inst.source if dataset_type.get_name() != MATRES_DATASET_NAME else inst.source.removeprefix('E')
        target = inst.target if dataset_type.get_name() != MATRES_DATASET_NAME else inst.target.removeprefix('E')
        label = inst.label
        gold_rels[f'{doc_id}#{source}#{target}'] = label

        if doc_id not in event_dict:
            event_dict[doc_id] = set()
        event_dict[doc_id].add(source)
        event_dict[doc_id].add(target)

    all_event_permut = {doc_id: list(permutations(event_set, 2)) for doc_id, event_set in event_dict.items()}
    all_posibile_keys = set()
    for doc_id, event_permut in all_event_permut.items():
        for event in event_permut:
            all_posibile_keys.add(f'{doc_id}#{event[0]}#{event[1]}')

    gold_rels_all = gold_rels.copy()
    aggregate_pred_as_dict = dict()
    for file_ in prediction_files:
        pred_as_dict, _ = read_pred_dot_file(file_, test_docs_dict, dataset_type)
        for key in all_posibile_keys:
            if key not in aggregate_pred_as_dict:
                if dataset_type.get_name() in [EVENTFULL_DATASET_NAME, MATRES_DATASET_NAME, NARRATIVE_4RELS_DATASET_NAME]:
                    aggregate_pred_as_dict[key] = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    aggregate_pred_as_dict[key] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            if key in pred_as_dict:
                aggregate_pred_as_dict[key][pred_as_dict[key]] += 1

            if key not in gold_rels_all:
                gold_rels_all[key] = 'NA'

    adjust_pred_and_rev_preds(aggregate_pred_as_dict, dataset_type.get_label_set())

    predictions = []
    order_list = []
    for key in gold_rels_all.keys():
        split = key.split("#")
        split.insert(2, gold_rels_all[key])
        order_list.append(split)
        if key in aggregate_pred_as_dict:
            probs = aggregate_pred_as_dict[key]
            _pred_norm = [list_itm / sum(probs) for list_itm in probs]
            predictions.append(_pred_norm)
        else:
            raise ValueError(f'Key-{key} not found in predictions!')

    np_predictions = np.array(predictions)
    all_golds, all_preds, gold_pred_mapping = run_transitive_constraints(np_predictions, order_list.copy(), None, alpha=-1,
                                                           dataset_type=dataset_type)

    pred_as_dict = dict()
    for doc_id, mapp_list in gold_pred_mapping.items():
        for key, item in mapp_list.items():
            pred_as_dict[f'{doc_id}#{key[0]}#{key[1]}'] = int(item[1])

    _, _, gold_for_trans, pred_for_trans, _ = convert_format(orig_ins_list, pred_as_dict, dataset_type.get_label_set(), debug=False)

    return all_golds, all_preds, gold_for_trans, pred_for_trans, pred_as_dict

def eval_full_doc(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type):
    final_golds = []
    final_preds = []
    for idx in range(len(all_golds)):
        if all_golds[idx] == -1:
            continue
        final_golds.append(all_golds[idx])
        final_preds.append(all_preds[idx])

    return evaluation(final_golds, final_preds, gold_for_trans, pred_for_trans, dataset_type)


def eval_by_sent(pred_as_dict, dataset_type, consecutive):
    final_golds = []
    final_preds = []
    for ins in _orig_ins_list:
        doc_id = ins.docid
        source = ins.source
        target = ins.target
        label = ins.label
        sentdiff = ins.sentdiff
        key = f'{doc_id}#{source}#{target}'
        rev_key = f'{doc_id}#{target}#{source}'

        condition_ = False
        if consecutive and sentdiff <= 1:
            condition_ = True
        elif not consecutive and sentdiff > 1:
            condition_ = True

        if condition_:
            if key in pred_as_dict:
                final_golds.append(_dataset_type.get_label_set()[label])
                final_preds.append(pred_as_dict[key])
            elif rev_key in pred_as_dict:
                final_golds.append(
                    _dataset_type.get_label_set()[_dataset_type.get_label_set().get_reverse_label(label)])
                final_preds.append(pred_as_dict[rev_key])
            else:
                raise KeyError(
                    f'Key {key} not found in aggregate predictions and reverse key {rev_key} not found in aggregate predictions!')

    return evaluation(final_golds, final_preds, None, None, dataset_type)



def gen_prediction_for_transitive(order_list, predictions, labels):
    pred_as_dict = {}
    pred_for_trans = {}
    gold_for_trans = {}
    for idx, item in enumerate(order_list):
        # key: doc_id#source_event#target_event, value: relation
        pred_as_dict[f'{item[0]}#{item[1]}#{item[3]}'] = predictions[idx]

        if item[0] not in pred_for_trans:
            pred_for_trans[item[0]] = []
            gold_for_trans[item[0]] = []
        pred_for_trans[item[0]].append((item[1], predictions[idx], item[3]))
        gold_for_trans[item[0]].append((item[1], labels.adjust_label(item[2]), item[3]))

    return gold_for_trans, pred_for_trans, pred_as_dict


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

    start_time = time.time()
    _all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _pred_as_dict = run_majority_vote_trans_const(_dataset_type, _test_docs_dict, _prediction_files, _orig_ins_list)
    end_time = time.time()

    print('\n\n####### Full Document Evaluation ####')
    f1_full = eval_full_doc(_all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _dataset_type)

    print('\n\n####### *Consecutive* Sentence Document Evaluation ####')
    f1_consec = eval_by_sent(_pred_as_dict, _dataset_type, True)
    print('\n\n####### *Non-Consecutive* Sentence Document Evaluation ####')
    f1_non_consec = eval_by_sent(_pred_as_dict, _dataset_type, False)
    execution_time = end_time - start_time

    print('\n\n###### Summary ######')
    print(f"Full F1: {f1_full}")
    print(f"Consecutive F1: {f1_consec}")
    print(f"Non-Consecutive F1: {f1_non_consec}")
    print(f"Execution time: {execution_time:.4f} seconds")
    print('Done!')
