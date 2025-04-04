import json

import numpy as np

from scripts.utils.classes.datasets_type import TBDDataset, DataType
from scripts.utils.classes.gurobi_optimizer import GurobiOptimizer


def run_transitive_constraints(predictions, order_list, output_file, alpha, dataset_type: DataType):
    labels = dataset_type.get_label_set()
    error_mat = np.zeros((labels.get_num_classes(), labels.get_num_classes()))
    all_doc_errors = dict()
    errors_log = []
    all_golds = []
    all_preds = []
    all_docs_gold_pred_mapping = dict()
    split_docs, all_nodes = split_to_docs(predictions, order_list, dataset_type)
    for doc_id, doc_preds in split_docs.items():
        optimizer = GurobiOptimizer(dataset_type=dataset_type, alpha=alpha)
        golds, preds, gold_pred_mapping = optimizer.init_and_run_constraints(doc_preds, all_nodes[doc_id])
        all_docs_gold_pred_mapping[doc_id] = gold_pred_mapping
        all_golds.extend(golds)
        all_preds.extend(preds)
        # doc_errors = optimizer.error_analysis(preds, golds, split_docs[doc_id], error_mat, labels.get_index_to_class())
        # all_doc_errors[doc_id] = len(doc_errors)
        # errors_log.extend(doc_errors)

    return all_golds, all_preds, all_docs_gold_pred_mapping


def split_to_docs(al_soft_probs, test_order_list, dataset_type: DataType):
    for i, pred in enumerate(al_soft_probs):
        test_order_list[i] = test_order_list[i] + [pred]

    split_docs = {}
    al_doc_ids = []
    all_nodes = {}
    label_set = dataset_type.get_label_set()
    for i, doc in enumerate(test_order_list):
        if doc[0] not in al_doc_ids:
            al_doc_ids.append(doc[0])
            all_nodes[doc[0]] = set()
            split_docs[doc[0]] = []
        # (Source_event, Target_event, Pred_Relation, Gold_Label, Doc_Id)
        split_docs[doc[0]].append((doc[1], doc[3], doc[4], label_set.adjust_label([doc[2]]), doc[0]))
        all_nodes[doc[0]].add(doc[1])
        all_nodes[doc[0]].add(doc[3])

    return split_docs, all_nodes
