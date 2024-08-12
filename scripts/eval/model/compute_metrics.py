from typing import Set, List, Dict


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


def set_edge_in_graph(graph_matrix, i, j, excepted_symmetric_rel):
    if graph_matrix[j][i] == "NA":
        graph_matrix[j][i] = excepted_symmetric_rel
    elif graph_matrix[j][i] != excepted_symmetric_rel:
        graph_matrix[i][j] = "contradict"
        graph_matrix[j][i] = "contradict"


def fill_transitive_closure(node_list: List, edge_list: List):
    graph_matrix, index_map = get_direct_reach_graph(node_list, edge_list)
    length = len(graph_matrix)

    for k in range(length):
        for i in range(length):
            for j in range(length):
                inferred_rel = get_transitive_relation(graph_matrix, i, j, k)
                empty_rel = graph_matrix[i][j] == "NA"

                if i == j or not empty_rel:
                    continue

                if inferred_rel == 'before':
                    graph_matrix[i][j] = 'before'
                    set_edge_in_graph(graph_matrix, i, j, 'after')
                elif inferred_rel == 'after':
                    graph_matrix[i][j] = 'after'
                    set_edge_in_graph(graph_matrix, i, j, 'before')
                elif inferred_rel == 'equal':
                    graph_matrix[i][j] = 'equal'
                    set_edge_in_graph(graph_matrix, i, j, 'equal')

    return from_matrix_to_edges(graph_matrix, index_map)


def from_matrix_to_edges(graph_matrix, index_map):
    edge_set = set()
    reversed_index_map = {v: k for k, v in index_map.items()}
    for i in range(len(graph_matrix)):
        for j in range(len(graph_matrix)):
            if graph_matrix[i][j] != "NA":
                edge_set.add((reversed_index_map[i], graph_matrix[i][j], reversed_index_map[j]))

    return edge_set


def get_direct_reach_graph(node_list: List, edge_list: List):
    graph_matrix = [["NA" for _ in range(len(node_list))] for _ in range(len(node_list))]
    index_map = {node: i for i, node in enumerate(node_list)}

    # Fill the direct edges
    for edge in edge_list:
        e1, rel, e2 = edge
        graph_matrix[index_map[e1]][index_map[e2]] = rel

    # Fill the missing symmetric edges
    for edge in edge_list:
        e1, rel, e2 = edge
        if rel == "equal":
            set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "equal")
        elif rel == "before":
            set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "after")
        elif rel == "after":
            set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "before")
        elif rel == 'vague':
            set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "vague")

    return graph_matrix, index_map


def fill_all_edges(edge_list: List, rel_dist_before: Dict):
    node_set = set()
    for edge in edge_list:
        node_set.add(edge[0])
        node_set.add(edge[2])
        rel_dist_before[edge[1]] = rel_dist_before.get(edge[1], 0) + 1

    edge_set = fill_transitive_closure(sorted(list(node_set)), edge_list)
    return node_set, edge_set


def calculate(test_dict):
    total_generated_nodes = 0
    total_gold_nodes = 0
    total_generated_edges = 0
    total_gold_edges = 0
    generated_edges_fill = 0
    gold_edges_fill = 0
    edges_filtered = 0

    node_precision = 0
    node_recall = 0

    edge_precision = 0
    edge_recall = 0

    gold_rel_dist_before = {}
    gold_rel_dist_after = {}
    gen_rel_dist_before = {}
    gen_rel_dist_after = {}
    currect_rel_dist = {}

    for doc_id in test_dict:
        consider_edges = set()
        gold_edge_list = test_dict[doc_id]['gold']
        total_gold_edges += len(gold_edge_list)

        generated_edge_list = test_dict[doc_id]['generated']
        total_generated_edges += len(generated_edge_list)

        gold_node_set, gold_edge_set = fill_all_edges(gold_edge_list, gold_rel_dist_before)
        for edge in gold_edge_set:
            gold_rel_dist_after[edge[1]] = gold_rel_dist_after.get(edge[1], 0) + 1

        gold_edges_fill += len(gold_edge_set)
        total_gold_nodes += len(gold_node_set)
        # After adding all possible edges, we can create the set of edges to consider
        for gold_edge in gold_edge_set:
            consider_edges.add(f'{gold_edge[0]}#{gold_edge[2]}')
            consider_edges.add(f'{gold_edge[2]}#{gold_edge[0]}')

        generated_node_set, generated_edge_set = fill_all_edges(generated_edge_list, gen_rel_dist_before)
        # After adding all possible generated edges, we can filter out the edges that are not in the gold graph
        generated_edge_set_filtered = set(filter(lambda x: f'{x[0]}#{x[2]}' in consider_edges, generated_edge_set))
        for edge in generated_edge_set_filtered:
            gen_rel_dist_after[edge[1]] = gen_rel_dist_after.get(edge[1], 0) + 1

        edges_filtered += (len(generated_edge_set) - len(generated_edge_set_filtered))
        generated_edges_fill += len(generated_edge_set_filtered)
        total_generated_nodes += len(generated_node_set)

        node_intersection = generated_node_set.intersection(gold_node_set)
        node_precision += len(node_intersection) / len(generated_node_set)
        node_recall += len(node_intersection) / len(gold_node_set)

        edge_intersection = generated_edge_set_filtered.intersection(gold_edge_set)
        edge_precision += len(edge_intersection) / len(generated_edge_set_filtered)
        edge_recall += len(edge_intersection) / len(gold_edge_set)
        for edge in edge_intersection:
            currect_rel_dist[edge[1]] = currect_rel_dist.get(edge[1], 0) + 1

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
    print("Edges Filtered (pairs no part of gold): ", edges_filtered)
    print("Gold Relation Distribution (before fill): ", gold_rel_dist_before)
    print("Gold Relation Distribution (after fill): ", gold_rel_dist_after)
    print("Generated Relation Distribution (before fill): ", gen_rel_dist_before)
    print("Generated Relation Distribution (after fill): ", gen_rel_dist_after)
    print("Correct Relation Distribution: ", currect_rel_dist)

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
