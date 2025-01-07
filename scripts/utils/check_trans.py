from collections import Counter

import numpy as np


MATRES_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}
Relations = {0: "Before", 1: "After", 2: "Equal", 3: "Vague"}


def get_symmetric_rel(rel):
    if rel == 0:
        return 1
    elif rel == 1:
        return 0
    else:
        return rel


def full_triplets_to_numpy_graph(triplet_list):
    event_ids = set()
    for triplet in triplet_list:
        if triplet[0] not in event_ids:
            event_ids.add(triplet[0])
        if triplet[2] not in event_ids:
            event_ids.add(triplet[2])

    event_ids_dict = {event: i for i, event in enumerate(sorted(list(event_ids)))}
    # build the graph
    graph = np.full((len(event_ids), len(event_ids)), -1)
    for triplet in triplet_list:
        graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = MATRES_labels[triplet[1].upper()]
        graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = get_symmetric_rel(MATRES_labels[triplet[1].upper()])

    # returning the upper triangular matrix
    return graph, event_ids_dict


def triplets_to_numpy_graph(triplet_list, is_pred):
    event_ids = set()
    for triplet in triplet_list:
        if triplet[0] not in event_ids:
            event_ids.add(triplet[0])
        if triplet[2] not in event_ids:
            event_ids.add(triplet[2])

    event_ids_dict = {event: i for i, event in enumerate(sorted(list(event_ids)))}
    # build the graph
    graph = np.full((len(event_ids), len(event_ids)), -1)
    rel_hist = {"Before": 0, "After": 0, "Equal": 0, "Vague": 0}
    for triplet in triplet_list:
        if is_pred:
            graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = triplet[1]
        else:
            if triplet[0] < triplet[2]:
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = triplet[1]
            else:
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = get_symmetric_rel(triplet[1])

        rel_hist[Relations[triplet[1]]] += 1

    # returning the upper triangular matrix
    return graph, event_ids_dict, rel_hist


def count_graph_transitive_discrepancies(graph, event_ids_dict):
    """
    Perform transitive closure of a directed acyclic graph (DAG)
    with relations 'before', 'after', 'equal', 'vague'.

    Parameters:
    graph (2D list or numpy array): Adjacency matrix of the input graph
                                    where each element is a string ('B', 'A', 'E', 'V', or '').
                                    '' means no relation.

    Returns:
    2D list: Adjacency matrix of the graph with transitive closure.
    """
    n = len(graph)
    closure_graph = np.copy(graph)
    error_log = []

    doc_contradictions = 0
    for i in range(n):
        for k in range(i, n):
            if k != i:
                if closure_graph[i][k] != get_symmetric_rel(closure_graph[k][i]):
                    doc_contradictions += 1

    event_ids_reversed = {v: k for k, v in event_ids_dict.items()}
    doc_discrepancies = 0
    for k in range(n):
        for i in range(n):
            if k != i:
                for j in range(i, n):
                    if i != j and j != k:
                        # If i -> k is before or equal and k -> j is 'before' relation
                        if closure_graph[i][k] == 0 and closure_graph[k][j] == 0:
                            if closure_graph[i][j] != 0:
                                doc_discrepancies += 1
                                error_log.append(f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]} == (Before, Before != Before)')
                        elif closure_graph[i][k] == 0 and closure_graph[k][j] == 2:
                            if closure_graph[i][j] != 0:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Before, Equal != Before)'))
                        elif closure_graph[i][k] == 2 and closure_graph[k][j] == 0:
                            if closure_graph[i][j] != 0:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Equal, Before != Before)'))
                        # If i -> k is 'after' or 'equal' and k -> j is after
                        elif closure_graph[i][k] == 1 and closure_graph[k][j] == 1:
                            if closure_graph[i][j] != 1:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(After, After != After)'))
                        elif closure_graph[i][k] == 1 and closure_graph[k][j] == 2:
                            if closure_graph[i][j] != 1:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(After, Equal != After)'))
                        elif closure_graph[i][k] == 2 and closure_graph[k][j] == 1:
                            if closure_graph[i][j] != 1:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Equal, After != After)'))
                        # If both i -> k and k -> j are equal
                        elif closure_graph[i][k] == 2 and closure_graph[k][j] == 2:
                            if closure_graph[i][j] != 2:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Equal, Equal != Equal)'))
                        # Handling interaction between 'vague' and other relations
                        elif closure_graph[i][k] == 0 and closure_graph[k][j] == 3:
                            if closure_graph[i][j] != 0 and closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Before, Vague != Before/Vague)'))
                        elif closure_graph[i][k] == 1 and closure_graph[k][j] == 3:
                            if closure_graph[i][j] != 1 and closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(After, Vague != After/Vague)'))
                        elif closure_graph[i][k] == 2 and closure_graph[k][j] == 3:
                            if closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Equal, Vague != Vague)'))
                        elif closure_graph[i][k] == 3 and closure_graph[k][j] == 2:
                            if closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Vague, Equal != Vague)'))
                        elif closure_graph[i][k] == 3 and closure_graph[k][j] == 0:
                            if closure_graph[i][j] != 0 and closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Vague, Before != Before/Vague)'))
                        elif closure_graph[i][k] == 3 and closure_graph[k][j] == 1:
                            if closure_graph[i][j] != 1 and closure_graph[i][j] != 3:
                                doc_discrepancies += 1
                                error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}', '(Vague, After != After/Vague)'))

    return doc_discrepancies, doc_contradictions, error_log


def transitive_closure_with_relations(graph):
    n = len(graph)
    closure_graph = np.copy(graph)

    for k in range(n):
        for i in range(n):
            if closure_graph[i][k] in (0, 1, 2):  # Skip vague relations
                for j in range(n):
                    if closure_graph[k][j] in (0, 1, 2):
                        # If i -> k is before or equal and k -> j is 'before' relation
                        if closure_graph[i][k] in [0, 2] and closure_graph[k][j] == 0:
                            if closure_graph[i][j] == -1:
                                closure_graph[i][j] = 0  # Add 'before'
                        # If i -> k is 'before' and k -> j is 'before' or 'equal'
                        elif closure_graph[i][k] == 0 and closure_graph[k][j] in [0, 2]:
                            if closure_graph[i][j] == -1:
                                closure_graph[i][j] = 0  # Add 'before' relation
                        # If i -> k is 'after' or 'equal' and k -> j is after
                        elif closure_graph[i][k] in [1, 2] and closure_graph[k][j] == 1:
                            if closure_graph[i][j] == -1:
                                closure_graph[i][j] = 1  # Add 'after' relation
                        # If i -> k is 'after' and k -> j is 'after' or 'equal'
                        elif closure_graph[i][k] == 1 and closure_graph[k][j] in [1, 2]:
                            if closure_graph[i][j] == -1:
                                closure_graph[i][j] = 1  # Add 'after' relation
                        # If both i -> k and k -> j are equal
                        elif closure_graph[i][k] == 2 and closure_graph[k][j] == 2:
                            if closure_graph[i][j] == -1:
                                closure_graph[i][j] = 2  # Add 'equal' relation

    return closure_graph


def from_graph_to_triplets(graph, event_ids_dict):
    triplets = []
    for i in range(len(graph)):
        for j in range(i, len(graph)):
            if j != i and graph[i][j] != -1:
                triplets.append((list(event_ids_dict.keys())[list(event_ids_dict.values()).index(i)], graph[i][j], list(event_ids_dict.keys())[list(event_ids_dict.values()).index(j)]))

    return triplets


def evaluate_triplets(doc_triplet, is_pred):
    doc_graph, event_ids_dict, hist = triplets_to_numpy_graph(doc_triplet, is_pred=is_pred)
    total_hist = Counter(hist)
    trans_discrepancies, sym_contradictions, error_log = count_graph_transitive_discrepancies(doc_graph, event_ids_dict)
    return total_hist, trans_discrepancies, sym_contradictions, error_log


def count_discrepancies(pred_triplet_dict, gold_triplet_dict, sym=False):
    total_trans_discrepancies = 0
    total_sym_contradictions = 0
    total_pred_hist = {"Before": 0, "After": 0, "Equal": 0, "Vague": 0}
    for doc_id in pred_triplet_dict:
        pred_hist, pred_trans_discrepancies, pred_sym_contradictions, _ = evaluate_triplets(pred_triplet_dict[doc_id], True)
        total_pred_hist = dict(Counter(total_pred_hist) + Counter(pred_hist))
        total_trans_discrepancies += pred_trans_discrepancies

        gold_hist, gold_trans_discrepancies, gold_sym_contradictions, error_log = evaluate_triplets(gold_triplet_dict[doc_id], False)

        if sym:
            total_sym_contradictions += pred_sym_contradictions
            # assert gold_sym_contradictions == 0.0

        # sanity check
        if gold_trans_discrepancies != 0:
            print('\n'.join(error_log))
            raise ValueError("Transitive discrepancies in gold graph")

    # print(json.dumps(pred_docs_discrepancies, indent=4))
    print(f"Trans-Discrepancies: {total_trans_discrepancies // 2}, Sym-Contradictions: {total_sym_contradictions // 2}, Total predictions: {total_pred_hist}")
