import json
from collections import Counter

from scripts.eval.dataset.utils import parse_DOT


def check_nodes_edges_count(graph):
    nodes = set()
    for edge in graph:
        nodes.add(edge[0])
        nodes.add(edge[2])

    graph_edge_num = len(graph)
    expected_edge_num = (pow(len(nodes), 2) - len(nodes)) / 2
    print(f"Node count: {len(nodes)}")
    print(f'Edge count: {graph_edge_num}')
    print(f'Expected edge count: {expected_edge_num}')
    print(f'Delta (expected-actual): {expected_edge_num - graph_edge_num}')

    duplicates = 0
    node_degree = dict()
    relation_distribution = dict()
    node_with_atleast_one_edge = set()
    key_set = set()
    for edge in graph:
        e1, rel, e2 = edge
        key = f'{e1}#{rel}#{e2}'
        if key in key_set:
            duplicates += 1
        else:
            node_degree[e1] = node_degree.get(e1, 0) + 1
            node_degree[e2] = node_degree.get(e2, 0) + 1
            node_with_atleast_one_edge.add(e1)
            node_with_atleast_one_edge.add(e2)
            relation_distribution[rel] = relation_distribution.get(rel, 0) + 1
            key_set.add(key)

    degree_hist_count = Counter(node_degree.values())
    print(f'Duplicates: {duplicates}')
    print(f'Avg Node degree: {sum(node_degree.values()) / len(node_degree.keys())}')
    print(f'Node degree distribution: {dict(sorted(dict(degree_hist_count).items()))}')
    print(f'Node without edges: {len(nodes) - len(node_with_atleast_one_edge)}')
    print(f'Relation distribution: {sorted(relation_distribution.items())}')
    return len(nodes), graph_edge_num, expected_edge_num, duplicates, degree_hist_count, relation_distribution


if __name__ == "__main__":
    # gold_file = "data/DOT_format/MATRES_test_dot.json"
    gold_file = "data/DOT_format/EventFull_test_dot.json"

    with open(gold_file) as f:
        golds = json.load(f)

    _tot_nodes = 0
    _graph_edge_num = 0
    _expected_edge_num = 0
    _duplicates = 0
    _degree_hist = Counter()
    _relation_distribution = Counter()
    for file in golds.keys():
        print(f'-------------- File: {file} ------------------')
        gold_graph, gold_duplicate = parse_DOT(golds[file]['target'])
        _nodes, _graph_edge_num, _expected_edge_num, _duplicates, _degree_hist_count, _relation_dist = check_nodes_edges_count(gold_graph)
        _tot_nodes += _nodes
        _graph_edge_num += _graph_edge_num
        _expected_edge_num += _expected_edge_num
        _duplicates += _duplicates
        _degree_hist += _degree_hist_count
        _relation_distribution += Counter(_relation_dist)
        print('---------------------------------------------')

    print('--------------- Dataset Stats------------------')
    print(f'Total nodes: {_tot_nodes}')
    print(f'Total edges: {_graph_edge_num}')
    print(f'Expected edges: {_expected_edge_num}')
    print(f'Duplicates: {_duplicates}')
    print(f'Avg Node degree: {sum(_degree_hist.keys()) / len(_degree_hist.keys())}')
    print(f'Node degree distribution (node_degree: num_of_such_nodes): {dict(sorted(dict(_degree_hist).items()))}')
    print(f'Relation distribution: {dict(sorted(dict(_relation_distribution).items()))}')

