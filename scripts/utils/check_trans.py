import copy
from collections import Counter


def triplets_to_numpy_graph(triplet_list, dataset_type, is_pred):
    label_set = dataset_type.get_label_set()
    event_ids = set()
    for triplet in triplet_list:
        if triplet[0] not in event_ids:
            event_ids.add(triplet[0])
        if triplet[2] not in event_ids:
            event_ids.add(triplet[2])

    event_ids_dict = {event: i for i, event in enumerate(sorted(list(event_ids)))}
    # build the graph
    graph = [['NA' for _ in range(len(event_ids))] for _ in range(len(event_ids))]
    rel_hist = dataset_type.get_label_set().get_labels_hist()

    index_to_class_lab = label_set.get_index_to_class()
    for triplet in triplet_list:
        if triplet[0] < triplet[2]:
            graph[event_ids_dict[triplet[0]]][event_ids_dict[triplet[2]]] = index_to_class_lab[triplet[1]]
        else:
            graph[event_ids_dict[triplet[2]]][event_ids_dict[triplet[0]]] = label_set.get_reverse_label(index_to_class_lab[triplet[1]])

        rel_hist[index_to_class_lab[triplet[1]]] += 1

    # returning the upper triangular matrix
    return graph, event_ids_dict, rel_hist


def count_graph_transitive_discrepancies_both(closure_graph, i, j, k):
    found = False
    if closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'BEFORE':
        if closure_graph[i][j] != 'BEFORE':
            found = True
    elif closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'EQUAL':
        if closure_graph[i][j] != 'BEFORE':
            found = True
    elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'BEFORE':
        if closure_graph[i][j] != 'BEFORE':
            found = True
    # If i -> k is 'after' or 'equal' and k -> j is after
    elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'AFTER':
        if closure_graph[i][j] != 'AFTER':
            found = True
    elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'EQUAL':
        if closure_graph[i][j] != 'AFTER':
            found = True
    elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'AFTER':
        if closure_graph[i][j] != 'AFTER':
            found = True
    # If both i -> k and k -> j are equal
    elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'EQUAL':
        if closure_graph[i][j] != 'EQUAL':
            found = True
    elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'VAGUE':
        if closure_graph[i][j] != 'VAGUE':
            found = True
    elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'EQUAL':
        if closure_graph[i][j] != 'VAGUE':
            found = True

    return found


def count_graph_transitive_discrepancies_4rels(graph, event_ids_dict, dataset_type):
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
    label_set = dataset_type.get_label_set()
    n = len(graph)
    closure_graph = copy.deepcopy(graph)
    error_log = []

    doc_contradictions = 0
    for i in range(n):
        for k in range(i, n):
            if k != i and closure_graph[k][i] != 'NA' and closure_graph[i][k] != 'NA':
                if closure_graph[i][k] != label_set.get_reverse_numerical_label(closure_graph[k][i]):
                    doc_contradictions += 1

    trans_set = set()
    pair_set = set()
    event_ids_reversed = {v: k for k, v in event_ids_dict.items()}
    doc_discrepancies = 0
    for k in range(n):
        for i in range(n):
            if k != i:
                for j in range(i, n):
                    found = False
                    node_set = tuple(sorted([i, j, k]))
                    pair = tuple(sorted([i, j]))
                    if node_set not in trans_set and pair not in pair_set:
                        if i != j and j != k and closure_graph[i][j] != 'NA' and closure_graph[i][k] != 'NA' and closure_graph[k][j] != 'NA':
                            # If i -> k is before or equal and k -> j is 'before' relation
                            if count_graph_transitive_discrepancies_both(closure_graph, i, j, k):
                                found = True
                            # Handling interaction between 'vague' and other relations
                            elif closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE':
                                    found = True
                            elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE':
                                    found = True
                            elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'BEFORE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE':
                                    found = True
                            elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'AFTER':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE':
                                    found = True

                        if found:
                            trans_set.add(node_set)
                            pair_set.add(pair)
                            doc_discrepancies += 1
                            error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}',
                                              f'({closure_graph[i][k]}, {closure_graph[k][j]} --> {closure_graph[i][j]})'))

    return doc_discrepancies, doc_contradictions, error_log


def count_graph_transitive_discrepancies_6rels(graph, event_ids_dict, dataset_type):
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
    label_set = dataset_type.get_label_set()
    n = len(graph)
    closure_graph = copy.deepcopy(graph)
    error_log = []

    doc_contradictions = 0
    for i in range(n):
        for k in range(i, n):
            if k != i and closure_graph[k][i] != 'NA' and closure_graph[i][k] != 'NA':
                if closure_graph[i][k] != label_set.get_reverse_numerical_label(closure_graph[k][i]):
                    doc_contradictions += 1

    trans_set = set()
    pair_set = set()
    event_ids_reversed = {v: k for k, v in event_ids_dict.items()}
    doc_discrepancies = 0
    for k in range(n):
        for i in range(n):
            if k != i:
                for j in range(i, n):
                    found = False
                    node_set = tuple(sorted([i, j, k]))
                    pair = tuple(sorted([i, j]))
                    if node_set not in trans_set and pair not in pair_set:
                        if i != j and j != k and closure_graph[i][j] != 'NA' and closure_graph[i][k] != 'NA' and closure_graph[k][j] != 'NA':
                            # If i -> k is before or equal and k -> j is 'before' relation
                            if count_graph_transitive_discrepancies_both(closure_graph, i, j, k):
                                found = True
                            elif closure_graph[i][k] == 'INCLUDES' and closure_graph[k][j] == 'INCLUDES':
                                if closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'INCLUDES' and closure_graph[k][j] == 'EQUAL':
                                if closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'INCLUDES':
                                if closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'IS_INCLUDED' and closure_graph[k][j] == 'IS_INCLUDED':
                                if closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'IS_INCLUDED' and closure_graph[k][j] == 'EQUAL':
                                if closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'EQUAL' and closure_graph[k][j] == 'IS_INCLUDED':
                                if closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True


                            elif closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES' and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                    and closure_graph[i][j] != 'INCLUDES' and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'BEFORE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES' and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'AFTER':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES' and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True

                            elif closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'INCLUDES':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'INCLUDES' and closure_graph[k][j] == 'BEFORE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                    and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'BEFORE' and closure_graph[k][j] == 'IS_INCLUDED':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'IS_INCLUDED' and closure_graph[k][j] == 'BEFORE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'VAGUE' \
                                    and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True

                            elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'INCLUDES':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'INCLUDES' and closure_graph[k][j] == 'AFTER':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'AFTER' and closure_graph[k][j] == 'IS_INCLUDED':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                            elif closure_graph[i][k] == 'IS_INCLUDED' and closure_graph[k][j] == 'AFTER':
                                if closure_graph[i][j] != 'AFTER' and closure_graph[i][j] != 'VAGUE' \
                                        and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True

                            elif closure_graph[i][k] == 'INCLUDES' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'AFTER' \
                                        and closure_graph[i][j] != 'VAGUE' and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'INCLUDES':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'AFTER' \
                                        and closure_graph[i][j] != 'VAGUE' and closure_graph[i][j] != 'INCLUDES':
                                    found = True
                            elif closure_graph[i][k] == 'IS_INCLUDED' and closure_graph[k][j] == 'VAGUE':
                                if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'AFTER' \
                                        and closure_graph[i][j] != 'VAGUE' and closure_graph[i][j] != 'IS_INCLUDED':
                                    found = True
                                elif closure_graph[i][k] == 'VAGUE' and closure_graph[k][j] == 'IS_INCLUDED':
                                    if closure_graph[i][j] != 'BEFORE' and closure_graph[i][j] != 'AFTER' \
                                            and closure_graph[i][j] != 'VAGUE' and closure_graph[i][j] != 'IS_INCLUDED':
                                        found = True

                        if found:
                            trans_set.add(node_set)
                            pair_set.add(pair)
                            doc_discrepancies += 1
                            error_log.append((f'{event_ids_reversed[i]}#{event_ids_reversed[k]}#{event_ids_reversed[j]}',
                                              f'({closure_graph[i][k]}, {closure_graph[k][j]} --> {closure_graph[i][j]})'))

    return doc_discrepancies, doc_contradictions, error_log


def from_graph_to_triplets(graph, event_ids_dict):
    triplets = []
    for i in range(len(graph)):
        for j in range(i, len(graph)):
            if j != i and graph[i][j] != -1:
                triplets.append((list(event_ids_dict.keys())[list(event_ids_dict.values()).index(i)], graph[i][j], list(event_ids_dict.keys())[list(event_ids_dict.values()).index(j)]))

    return triplets


def evaluate_triplets(doc_triplet, dataset_type, is_pred):
    doc_graph, event_ids_dict, hist = triplets_to_numpy_graph(doc_triplet, dataset_type, is_pred=is_pred)
    total_hist = Counter(hist)

    if dataset_type.get_label_set().get_num_classes() == 4:
        trans_discrepancies, sym_contradictions, error_log = count_graph_transitive_discrepancies_4rels(doc_graph, event_ids_dict, dataset_type)
    elif dataset_type.get_label_set().get_num_classes() == 6:
        trans_discrepancies, sym_contradictions, error_log = count_graph_transitive_discrepancies_6rels(doc_graph, event_ids_dict, dataset_type)
    else:
        raise ValueError("Unsupported number of classes in the dataset")

    return total_hist, trans_discrepancies, sym_contradictions, error_log


def count_discrepancies(pred_triplet_dict, gold_triplet_dict, dataset_type, sym=False):
    total_trans_discrepancies = 0
    total_sym_contradictions = 0
    total_pred_hist = dataset_type.get_label_set().get_labels_hist()
    total_gold_discrepancies = 0
    for doc_id in pred_triplet_dict:
        pred_hist, pred_trans_discrepancies, pred_sym_contradictions, pred_error_log = evaluate_triplets(pred_triplet_dict[doc_id], dataset_type, True)
        total_pred_hist = dict(Counter(total_pred_hist) + Counter(pred_hist))
        total_trans_discrepancies += pred_trans_discrepancies

        gold_hist, gold_trans_discrepancies, gold_sym_contradictions, error_log = evaluate_triplets(gold_triplet_dict[doc_id], dataset_type, False)

        if sym:
            total_sym_contradictions += pred_sym_contradictions
            # assert gold_sym_contradictions == 0.0

        # sanity check
        if gold_trans_discrepancies != 0:
            for pair in error_log:
                print(pair)
            total_gold_discrepancies += gold_trans_discrepancies

    print(f"Found {total_gold_discrepancies} transitive discrepancies in gold graph")
    # print(json.dumps(pred_docs_discrepancies, indent=4))
    print(f"Total-Docs: {len(gold_triplet_dict)}")
    print(f"Trans-Discrepancies: {total_trans_discrepancies}, Sym-Contradictions: {total_sym_contradictions // 2}, Total predictions: {total_pred_hist}")
