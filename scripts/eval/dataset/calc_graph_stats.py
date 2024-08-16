import json
from collections import Counter

from scripts.eval.dataset.utils import parse_DOT
from scripts.eval.model.eval_obj import EvalObj


def check_nodes_edges_count(eval_obj: EvalObj):
    expected_edge_num = (pow(len(eval_obj.node_set), 2) - len(eval_obj.node_set)) / 2
    print(f"Node count: {len(eval_obj.node_set)}")
    print(f'Edge count: {len(eval_obj.edge_set)}')
    print(f'Expected edge count: {expected_edge_num}')
    print(f'Delta (expected-actual): {expected_edge_num - len(eval_obj.edge_set)}')

    degree_hist_count = Counter(eval_obj.orig_node_degree.values())
    print(f'Duplicates: {eval_obj.duplicates}')
    print(f'Direct Contradictions: {eval_obj.contradictions}')
    print(f'Avg Node degree: {sum(eval_obj.orig_node_degree.values()) / len(eval_obj.orig_node_degree.keys())}')
    print(f'Node degree distribution before edge align: {dict(sorted(dict(eval_obj.orig_distribution).items()))}')
    print(f'Node degree distribution after edge align: {dict(sorted(dict(eval_obj.set_distribution).items()))}')
    print(f'Node without edges: {len(eval_obj.node_set) - len(eval_obj.orig_node_degree)}')
    print(f'Original relation distribution: {sorted(eval_obj.orig_distribution.items())}')
    print(f'Relation distribution: {sorted(eval_obj.set_distribution.items())}')

    return expected_edge_num, degree_hist_count


def calc_stats(golds):
    _tot_nodes = 0
    _graph_edge_num = 0
    _expected_edge_num = 0
    _duplicates = 0
    _direct_contradictions = 0
    _degree_hist = Counter()
    _aligned_rel_dist = Counter()
    _orig_rel_dist = Counter()
    for file in golds.keys():
        print(f'-------------- File: {file} ------------------')
        gold_graph = parse_DOT(golds[file]['target'])
        if gold_graph is None:
            print(f'Error in parsing {file}')
            continue

        _eval_obj = EvalObj(file, gold_graph)
        _expected_edges, _degree_hist_count = check_nodes_edges_count(_eval_obj)
        _tot_nodes += len(_eval_obj.node_set)
        _graph_edge_num += len(_eval_obj.edge_set)
        _expected_edge_num += _expected_edges
        _duplicates += _eval_obj.duplicates
        _degree_hist += _degree_hist_count
        _orig_rel_dist += Counter(_eval_obj.orig_distribution)
        _aligned_rel_dist += Counter(_eval_obj.set_distribution)
        _direct_contradictions += _eval_obj.contradictions
        print('---------------------------------------------')

    print('--------------- Dataset Stats------------------')
    print(f'Total nodes: {_tot_nodes}')
    print(f'Total (aligned) edges: {_graph_edge_num}')
    print(f'Expected edges: {_expected_edge_num}')
    print(f'Duplicates: {_duplicates}')
    print(f'Direct Contradictions: {_direct_contradictions}')
    print(f'Avg Node degree: {sum(_degree_hist.keys()) / len(_degree_hist.keys())}')
    print(f'Node degree distribution (node_degree: num_of_such_nodes): {dict(sorted(dict(_degree_hist).items()))}')
    print(f'Original Relation distribution: {dict(sorted(dict(_orig_rel_dist).items()))}, tot={sum(_orig_rel_dist.values())}')
    print(f'After Alignment Relation distribution: {dict(sorted(dict(_aligned_rel_dist).items()))}')


if __name__ == "__main__":
    in_file = "data/DOT_format/MATRES_train_dot.json"
    # in_file = "data/DOT_format/EventFull_test_dot.json"
    # in_file = "data/my_data/predictions/output/experiments/matres/eventfull_run_gpt3_5_-1pred_1exmples_task_description_v2.json"

    with open(in_file) as f:
        _golds = json.load(f)

    calc_stats(_golds)
