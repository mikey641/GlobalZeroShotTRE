from collections import Counter
from enum import Enum

from scripts.eval.model.eval_obj import EvalObj


class DIST_TYPE(Enum):
    NOT_ALIGNED = 1
    ALIGNED = 2
    REDUCED = 3


def get_dist(eval_objs, dist_type=DIST_TYPE.NOT_ALIGNED):
    if dist_type == DIST_TYPE.NOT_ALIGNED:
        dict_sum = sum([Counter(eval_objs[doc_id].orig_distribution) for doc_id in eval_objs], Counter())
    elif dist_type == DIST_TYPE.ALIGNED:
        dict_sum = sum([Counter(eval_objs[doc_id].set_distribution) for doc_id in eval_objs], Counter())
    elif dist_type == DIST_TYPE.REDUCED:
        dict_sum = sum([Counter(eval_objs[doc_id].reduced_distribution) for doc_id in eval_objs], Counter())
    else:
        raise ValueError("Invalid dist type")

    return sorted(dict(dict_sum).items()), sum(dict_sum.values())


def calculate(test_dict):
    edges_filtered = 0
    true_positive_edges = []

    node_precision = 0
    node_recall = 0

    edge_precision = 0
    edge_recall = 0

    gold_eval_objs = dict()
    pred_eval_objs = dict()

    print("##### Per Doc Precision and Recall #####")
    for doc_id in test_dict:
        print("Doc ID: ", doc_id)
        gold_edge_list = test_dict[doc_id]['gold']
        gold_eval_obj = EvalObj(doc_id, gold_edge_list)
        gold_eval_objs[doc_id] = gold_eval_obj

        gen_edge_list = test_dict[doc_id]['generated']
        gen_eval_obj = EvalObj(doc_id, gen_edge_list)
        pred_eval_objs[doc_id] = gen_eval_obj

        node_intersection = gen_eval_obj.node_set.intersection(gold_eval_obj.node_set)
        node_precision += len(node_intersection) / len(gen_eval_obj.node_set)
        node_recall += len(node_intersection) / len(gold_eval_obj.node_set)

        edge_intersection = gen_eval_obj.edge_set.intersection(gold_eval_obj.edge_set)
        true_positive_edges.append(EvalObj.calc_edge_distributions(edge_intersection))
        e_per = len(edge_intersection) / (len(gen_eval_obj.edge_set))
        e_rec = len(edge_intersection) / len(gold_eval_obj.edge_set)
        edge_precision += e_per
        edge_recall += e_rec
        print("Edge Precision: ", e_per)
        print("Edge Recall: ", e_rec)
        print("Edge F1: ", 2 * e_per * e_rec / (e_per + e_rec))
        print("=====================================")
        # edges_filtered += edge_filtered

    print("---------------------Nodes---------------------")
    node_precision = node_precision / len(test_dict)
    node_recall = node_recall / len(test_dict)
    print("----States-----")
    print("Gold Nodes: ", sum([len(gold_eval_objs[doc_id].node_set) for doc_id in gold_eval_objs]))
    print("Predicted Nodes: ", sum([len(pred_eval_objs[doc_id].node_set) for doc_id in pred_eval_objs]))
    print("----Eval-----")
    print("Node Precision: ", node_precision)
    print("Node Recall: ", node_recall)
    print("Node F1: ", 2 * node_precision * node_recall / (node_precision + node_recall))

    print("---------------------Edges---------------------")
    edge_precision = edge_precision / len(test_dict)
    edge_recall = edge_recall / len(test_dict)

    print("----States-----")
    gold_dist_before, gold_dist_before_sanity = get_dist(gold_eval_objs, DIST_TYPE.NOT_ALIGNED)
    gold_dist_after, gold_dist_after_sanity = get_dist(gold_eval_objs, DIST_TYPE.ALIGNED)
    # gold_dist_reduced, gold_dist_reduced_sanity = get_dist(gold_eval_objs, DIST_TYPE.REDUCED)
    pred_dist_before, pred_dist_before_sanity = get_dist(pred_eval_objs, DIST_TYPE.NOT_ALIGNED)
    pred_dist_after, pred_dist_after_sanity = get_dist(pred_eval_objs, DIST_TYPE.ALIGNED)
    # pred_dist_reduced, pred_dist_reduced_sanity = get_dist(pred_eval_objs, DIST_TYPE.REDUCED)

    true_positive_edges_dist = sorted(dict(sum([Counter(d) for d in true_positive_edges], Counter())).items())
    true_positive_edges_dist_tot = sum(dict(true_positive_edges_dist).values())

    print("### GLOD ###")
    print("Gold Original Edges (before align): ", sum([len(gold_eval_objs[doc_id].orig_edge_list) for doc_id in gold_eval_objs]))
    print("Gold Total Edges (after align complete): ", sum([len(gold_eval_objs[doc_id].edge_set) for doc_id in gold_eval_objs]))
    # print("Gold Total Edges (after align reduced): ", sum([len(gold_eval_objs[doc_id].edge_set_reduced) for doc_id in gold_eval_objs]))
    print(f"Gold Original Edges Distribution (before): {gold_dist_before}, tot={gold_dist_before_sanity}")
    print(f"Gold Total Edges Distribution (after): {gold_dist_after}, tot={gold_dist_after_sanity}")
    # print(f"Gold Total Edges Distribution (reduced): {gold_dist_reduced}, tot={gold_dist_reduced_sanity}")
    print()

    print("\n### PRED ###")
    print("Predicted Edges (before align): ", sum([len(pred_eval_objs[doc_id].orig_edge_list) for doc_id in pred_eval_objs]))
    print("Predicted Edges (after align complete): ", sum([len(pred_eval_objs[doc_id].edge_set) for doc_id in pred_eval_objs]))
    # print("Predicted Edges (after align reduced): ", sum([len(pred_eval_objs[doc_id].edge_set_reduced) for doc_id in pred_eval_objs]))
    print(f"Predicted Edges Distribution (before): {pred_dist_before}, tot={pred_dist_before_sanity}")
    print(f"Predicted Edges Distribution (after): {pred_dist_after}, tot={pred_dist_after_sanity}")
    # print(f"Predicted Edges Distribution (reduced): {pred_dist_reduced}, tot={pred_dist_reduced_sanity}")
    print(f"Predicted Edges Direct Duplicate: {sum([pred_eval_objs[doc_id].duplicates for doc_id in pred_eval_objs])}")
    print(f"Predicted Edges Direct Contradictions: {sum([pred_eval_objs[doc_id].contradictions for doc_id in pred_eval_objs])}")
    print("Predicted Edges Filtered (edges not part of gold): ", edges_filtered)
    print(f"Edge Intersect Distribution: {true_positive_edges_dist}, tot={true_positive_edges_dist_tot}")

    print("\n----Eval-----")
    print("Edge Precision: ", edge_precision)
    print("Edge Recall: ", edge_recall)
    print("Edge F1: ", 2 * edge_precision * edge_recall / (edge_precision + edge_recall))


if __name__ == "__main__":
    in_list = [("e1", "before", "e2"), ("e2", "before", "e3")]
    expected_edges = [
        ("e1", "before", "e2"),
        ("e1", "before", "e3"),
        ("e2", "before", "e3"),
    ]

    expected_nodes = {"e1", "e2", "e3"}

    eval_obj = EvalObj(None, in_list)
    assert eval_obj.node_set == expected_nodes
    assert eval_obj.edge_set == set(expected_edges)

    in_list = [("e1", "equal", "e2"), ("e2", "before", "e3")]
    expected_edges = [
        ("e1", "equal", "e2"),
        ("e2", "before", "e3"),
        ("e1", "before", "e3"),
    ]

    eval_obj = EvalObj(None, in_list)
    assert eval_obj.node_set == expected_nodes
    assert eval_obj.edge_set == set(expected_edges)
