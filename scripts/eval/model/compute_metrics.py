from typing import Set, Tuple, List, Dict


def get_transitive_relation(reach_graph, i, j, k):
    if ((reach_graph[i][k] == 'after' and reach_graph[k][j] == 'after') or
            (reach_graph[i][k] == 'after' and reach_graph[k][j] == 'equal') or
            (reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'after')):
        return 'after'
    elif ((reach_graph[i][k] == 'before' and reach_graph[k][j] == 'before') or
          (reach_graph[i][k] == 'before' and reach_graph[k][j] == 'equal') or
          (reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'before')):
        return 'before'
    elif reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'equal':
        return 'equal'
    else:
        return 'NA'


def fill_transitive_closure(node_set: Set, edge_set: Set):
    graph_matrix, index_map = get_direct_reach_graph(node_set, edge_set)
    index_map_reverse = {v: k for k, v in index_map.items()}
    length = len(graph_matrix)

    for k in range(length):
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue

                inferred_rel = get_transitive_relation(graph_matrix, i, j, k)
                empty_rel = graph_matrix[i][j] == "NA"
                if inferred_rel == 'before':
                    if empty_rel:
                        graph_matrix[i][j] = 'before'
                        graph_matrix[j][i] = 'after'
                        edge_set.add((index_map_reverse[i], 'before', index_map_reverse[j]))
                        edge_set.add((index_map_reverse[j], 'after', index_map_reverse[i]))
                elif inferred_rel == 'after':
                    if empty_rel:
                        graph_matrix[i][j] = 'after'
                        graph_matrix[j][i] = 'before'
                        edge_set.add((index_map_reverse[i], 'after', index_map_reverse[j]))
                        edge_set.add((index_map_reverse[j], 'before', index_map_reverse[i]))
                elif inferred_rel == 'equal':
                    if empty_rel:
                        graph_matrix[i][j] = 'equal'
                        graph_matrix[j][i] = 'equal'
                        edge_set.add((index_map_reverse[i], 'equal', index_map_reverse[j]))
                        edge_set.add((index_map_reverse[j], 'equal', index_map_reverse[i]))


def get_direct_reach_graph(node_set: Set, edge_set: Set):
    graph_matrix = [["NA" for _ in range(len(node_set))] for _ in range(len(node_set))]
    index_map = {node: i for i, node in enumerate(node_set)}

    for edge in edge_set:
        e1, rel, e2 = edge
        if rel == "equal":
            graph_matrix[index_map[e1]][index_map[e2]] = "equal"
            graph_matrix[index_map[e2]][index_map[e1]] = "equal"
        elif rel == "before":
            graph_matrix[index_map[e1]][index_map[e2]] = "before"
            graph_matrix[index_map[e2]][index_map[e1]] = "after"
        elif rel == "after":
            graph_matrix[index_map[e1]][index_map[e2]] = "after"
            graph_matrix[index_map[e2]][index_map[e1]] = "before"
        elif rel == 'vague':
            graph_matrix[index_map[e1]][index_map[e2]] = "vague"
            graph_matrix[index_map[e2]][index_map[e1]] = "vague"

    return graph_matrix, index_map


def add_edge_to_set(edge_set: Set, edge: Tuple):
    e1, rel, e2 = edge
    if rel == "equal":
        edge_set.add((e1, rel, e2))
        edge_set.add((e2, rel, e1))
    elif rel == "before":
        edge_set.add((e1, rel, e2))
        edge_set.add((e2, "after", e1))
    elif rel == "after":
        edge_set.add((e1, rel, e2))
        edge_set.add((e2, "before", e1))
    elif rel == 'vague':
        edge_set.add((e1, rel, e2))
        edge_set.add((e2, rel, e1))


def fill_all_edges(edge_list: List, rel_dist: Dict):
    node_set = set()
    edge_set = set()
    for edge in edge_list:
        node_set.add(edge[0])
        node_set.add(edge[2])
        add_edge_to_set(edge_set, edge)
        rel_dist[edge[1]] = rel_dist.get(edge[1], 0) + 1

    fill_transitive_closure(node_set, edge_set)
    return node_set, edge_set


def calculate(test_dict):
    total_generated_nodes = 0
    total_gold_nodes = 0
    total_generated_edges = 0
    total_gold_edges = 0
    generated_edges_fill = 0
    gold_edges_fill = 0

    node_precision = 0
    node_recall = 0

    edge_precision = 0
    edge_recall = 0

    gold_rel_dist = {}
    gen_rel_dist = {}

    for doc_id in test_dict:
        consider_edges = set()
        gold_edge_list = test_dict[doc_id]['gold']
        total_gold_edges += len(gold_edge_list)

        generated_edge_list = test_dict[doc_id]['generated']
        total_generated_edges += len(generated_edge_list)

        gold_node_set, gold_edge_set = fill_all_edges(gold_edge_list, gold_rel_dist)
        gold_edges_fill += len(gold_edge_set)
        total_gold_nodes += len(gold_node_set)
        # After adding all possible edges, we can create the set of edges to consider
        for gold_edge in gold_edge_set:
            consider_edges.add(f'{gold_edge[0]}, {gold_edge[2]}')
            consider_edges.add(f'{gold_edge[2]}, {gold_edge[0]}')

        generated_node_set, generated_edge_set = fill_all_edges(generated_edge_list, gen_rel_dist)
        # After adding all possible generated edges, we can filter out the edges that are not in the gold graph
        generated_edge_set = set(filter(lambda x: f'{x[0]}, {x[2]}' in consider_edges, generated_edge_set))
        generated_edges_fill += len(generated_edge_set)
        total_generated_nodes += len(generated_node_set)

        node_intersection = generated_node_set.intersection(gold_node_set)
        node_precision += len(node_intersection) / len(generated_node_set)
        node_recall += len(node_intersection) / len(gold_node_set)

        edge_intersection = generated_edge_set.intersection(gold_edge_set)
        edge_precision += len(edge_intersection) / len(generated_edge_set)
        edge_recall += len(edge_intersection) / len(gold_edge_set)

    print("---------------------Nodes---------------------")
    node_precision = node_precision / len(test_dict)
    node_recall = node_recall / len(test_dict)
    print("----States-----")
    print("Gold Total Nodes: ", total_gold_nodes)
    print("Generated Total Nodes: ", total_generated_nodes)
    print("----Eval-----")
    print("Node Precision: ", node_precision)
    print("Node Recall: ", node_recall)
    print("Node F1: ", 2 * node_precision * node_recall / (node_precision + node_recall))

    print("---------------------Edges---------------------")
    edge_precision = edge_precision / len(test_dict)
    edge_recall = edge_recall / len(test_dict)
    print("----States-----")
    print("Gold Total Edges (before fill): ", total_gold_edges)
    print("Gold Total Edges (after fill): ", gold_edges_fill)
    print("Generated Total Edges (before fill): ", total_generated_edges)
    print("Generated Total Edges (after fill): ", generated_edges_fill)
    print("Gold Relation Distribution: ", gold_rel_dist)
    print("Generated Relation Distribution: ", gen_rel_dist)

    print("----Eval-----")
    print("Edge Precision: ", edge_precision)
    print("Edge Recall: ", edge_recall)
    print("Edge F1: ", 2 * edge_precision * edge_recall / (edge_precision + edge_recall))


if __name__ == "__main__":
    in_list = [("e1", "before", "e2"), ("e2", "before", "e3")]
    expected_edges = [
        ("e1", "before", "e2"),
        ("e2", "after", "e1"),
        ("e2", "before", "e3"),
        ("e3", "after", "e2"),
        ("e1", "before", "e3"),
        ("e3", "after", "e1")
    ]

    expected_nodes = {"e1", "e2", "e3"}

    _node_set, _edge_set = fill_all_edges(in_list, {})
    assert _node_set == expected_nodes
    assert _edge_set == set(expected_edges)

    in_list = [("e1", "equal", "e2"), ("e2", "before", "e3")]
    expected_edges = [
        ("e1", "equal", "e2"),
        ("e2", "equal", "e1"),
        ("e2", "before", "e3"),
        ("e3", "after", "e2"),
        ("e1", "before", "e3"),
        ("e3", "after", "e1")
    ]

    _node_set, _edge_set = fill_all_edges(in_list, {})
    assert _node_set == expected_nodes
    assert _edge_set == set(expected_edges)
