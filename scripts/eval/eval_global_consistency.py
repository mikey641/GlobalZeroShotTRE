import argparse
import os
import time
from itertools import permutations

import numpy as np

from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation
from scripts.eval.shared.gurobi_optimizer_entrop import run_transitive_constraints
from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, OMNITEMP_DATASET_NAME, MavenDataset, \
    MAVEN_DATASET_NAME, OmniTempDataset, NarrativeDataset, MatresDataset, TBDDataset
from scripts.utils.io_utils import read_pred_dot_file, load_golds


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


def run_majority_vote_trans_const(dataset_type, test_as_dict, all_gold_files, prediction_files):
    gold_rels = dict()
    event_dict = dict()
    for key in test_as_dict.keys():
        doc_id, source, target = key.split('#')

        if doc_id not in event_dict:
            event_dict[doc_id] = set()
        event_dict[doc_id].add(source)
        event_dict[doc_id].add(target)
        gold_rels[f'{doc_id}#{source}#{target}'] = dataset_type.get_label_set().get_index_to_class()[test_as_dict[key]]

    all_event_permut = {doc_id: list(permutations(event_set, 2)) for doc_id, event_set in event_dict.items()}
    all_posibile_keys = set()
    for doc_id, event_permut in all_event_permut.items():
        for event in event_permut:
            all_posibile_keys.add(f'{doc_id}#{event[0]}#{event[1]}')

    aggregate_pred_as_dict = dict()
    for file_ in prediction_files:
        pred_as_dict, _ = read_pred_dot_file(file_, all_gold_files, dataset_type)
        for key in all_posibile_keys:
            if key not in aggregate_pred_as_dict:
                if dataset_type.get_name() in [OMNITEMP_DATASET_NAME, MATRES_DATASET_NAME, MAVEN_DATASET_NAME]:
                    aggregate_pred_as_dict[key] = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    aggregate_pred_as_dict[key] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            if key in pred_as_dict:
                aggregate_pred_as_dict[key][pred_as_dict[key]] += 1

            if key not in gold_rels:
                gold_rels[key] = 'NA'

    adjust_pred_and_rev_preds(aggregate_pred_as_dict, dataset_type.get_label_set())

    predictions = []
    order_list = []
    for key in gold_rels.keys():
        split = key.split("#")
        split.insert(2, gold_rels[key])
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

    _, _, gold_for_trans, pred_for_trans, _ = convert_format(test_as_dict, pred_as_dict, dataset_type.get_label_set(), debug=False)

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

    _test_as_dict, _all_gold_files = load_golds(_dataset_type.get_test_file(), _dataset_type.get_label_set())

    start_time = time.time()
    _all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _pred_as_dict = run_majority_vote_trans_const(_dataset_type, _test_as_dict, _all_gold_files, _prediction_files)
    end_time = time.time()
    print('\n\n####### Full Document Evaluation ####')
    f1_full = eval_full_doc(_all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _dataset_type)
    execution_time = end_time - start_time

    print('\n\n###### Summary ######')
    print(f"Full F1: {f1_full}")
    print(f"Execution time: {execution_time:.4f} seconds")
    print('Done!')
