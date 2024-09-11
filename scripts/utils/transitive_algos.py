import numpy
import numpy as np


def transitive_closure_with_relations(graph):
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

    for k in range(n):
        for i in range(n):
            if closure_graph[i][k] in ('B', 'A', 'E'):  # Skip vague relations
                for j in range(n):
                    if closure_graph[k][j] in ('B', 'A', 'E'):
                        # If i -> k is before or equal and k -> j is 'before' relation
                        if closure_graph[i][k] in ['B', 'E'] and closure_graph[k][j] == 'B':
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'B'  # Add 'before'
                        # If i -> k is 'before' and k -> j is 'before' or 'equal'
                        elif closure_graph[i][k] == 'B' and closure_graph[k][j] in ['B', 'E']:
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'B'  # Add 'before' relation
                        # If i -> k is 'after' or 'equal' and k -> j is after
                        elif closure_graph[i][k] in ['A', 'E'] and closure_graph[k][j] == 'A':
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'A'  # Add 'after' relation
                        # If i -> k is 'after' and k -> j is 'after' or 'equal'
                        elif closure_graph[i][k] == 'A' and closure_graph[k][j] in ['A', 'E']:
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'A'  # Add 'after' relation
                        # If both i -> k and k -> j are equal
                        elif closure_graph[i][k] == 'E' and closure_graph[k][j] == 'E':
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'E'  # Add 'equal' relation
                        # Handling interaction between 'vague' and other relations
                        if closure_graph[i][k] == 'V' or closure_graph[k][j] == 'V':
                            if closure_graph[i][j] == '':
                                closure_graph[i][j] = 'V'  # Add 'vague' relation
                        # Handle mixed relation cases (e.g., before and after)
                        # You can define custom behavior here for how different relations interact

    return closure_graph


def transitive_reduction_with_relations(graph):
    """
    Perform transitive reduction of a directed acyclic graph (DAG)
    with relations 'before', 'after', 'equal', 'vague'.

    Parameters:
    B=before, A=after, E=equal, V=vague, ''=no relation
    graph (2D list or numpy array): Adjacency matrix of the input graph
                                    where each element is a string ('B', 'A', 'E', 'V', or '').
                                    '' means no relation.

    Returns:
    2D list: Adjacency matrix of the reduced graph.
    """
    n = len(graph)
    reduced_graph = np.copy(graph)
    total_reducted = 0
    for k in range(n):
        for i in range(n):
            if reduced_graph[i][k] in ('B', 'A', 'E'):  # Skip vague relations
                for j in range(n):
                    if reduced_graph[k][j] in ('B', 'A', 'E'):
                        # If i -> k is 'before' or 'equal' and k -> j is 'before', infer i -> j is 'before'
                        if reduced_graph[i][k] in ['B', 'E'] and reduced_graph[k][j] == 'B':
                            if reduced_graph[i][j] == 'B':
                                reduced_graph[i][j] = ''  # Remove redundant 'before'
                                total_reducted += 1
                        # If i -> k is 'before' and k -> j is 'before' or 'equal', infer i -> j is 'before'
                        elif reduced_graph[i][k] == 'B' and reduced_graph[k][j] in ['B', 'E']:
                            if reduced_graph[i][j] == 'B':
                                reduced_graph[i][j] = ''  # Remove redundant 'before'
                                total_reducted += 1
                        # If i -> k is 'after' or 'equal' and k -> j is 'after', infer i -> j is 'after'
                        elif reduced_graph[i][k] == ['A', 'E'] and reduced_graph[k][j] == 'A':
                            if reduced_graph[i][j] == 'A':
                                reduced_graph[i][j] = ''  # Remove redundant 'after'
                                total_reducted += 1
                        # If i -> k is 'after' and k -> j is 'after' or 'equal', infer i -> j is 'after'
                        elif reduced_graph[i][k] == 'A' and reduced_graph[k][j] in ['A', 'E']:
                            if reduced_graph[i][j] == 'A':
                                reduced_graph[i][j] = ''  # Remove redundant 'after'
                                total_reducted += 1
                        # If i -> k is 'equal' and k -> j is 'equal', infer i -> j is 'equal'
                        elif reduced_graph[i][k] == 'E' and reduced_graph[k][j] == 'E':
                            if reduced_graph[i][j] == 'E':
                                reduced_graph[i][j] = ''  # Remove redundant 'equal'
                                total_reducted += 1
                        # Handle other combinations if needed (e.g., 'vague' or mismatching relations)

    return reduced_graph, total_reducted


# This method is for converting from the json format to numpy graph
def json_to_numpy_graph(content):
    event_ids = dict()
    running_ids = 0
    for pair in content['pairs']:
        if pair['_firstId'] not in event_ids:
            event_ids[pair['_firstId']] = running_ids
            running_ids += 1
        if pair['_secondId'] not in event_ids:
            event_ids[pair['_secondId']] = running_ids
            running_ids += 1

    # build the graph
    graph = np.full((running_ids, running_ids), '')
    for pair in content['pairs']:
        if pair['_relation'] == 'before':
            graph[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] = 'B'
        elif pair['_relation'] == 'after':
            graph[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] = 'A'
        elif pair['_relation'] == 'equal':
            graph[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] = 'E'
        elif pair['_relation'] == 'uncertain':
            graph[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] = 'V'

    return graph, event_ids


# This method is for the evaluation script
def triplets_to_numpy_graph(triplet_list):
    event_ids = set()
    for triplet in triplet_list:
        if triplet[0] not in event_ids:
            event_ids.add(triplet[0])
        if triplet[2] not in event_ids:
            event_ids.add(triplet[2])

    event_ids_dict = {event: i for i, event in enumerate(sorted(list(event_ids)))}
    # build the graph
    graph = np.full((len(event_ids), len(event_ids)), '')
    for triplet in triplet_list:
        if triplet[1] == 'before':
            if graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == '' or graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == 'B':
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'B'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'A'
            else:
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'C'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'C'
        elif triplet[1] == 'after':
            if graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == '' or graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == 'A':
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'A'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'B'
            else:
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'C'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'C'
        elif triplet[1] == 'equal':
            if graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == '' or graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == 'E':
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'E'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'E'
            else:
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'C'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'C'
        elif triplet[1] == 'vague':
            if graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == '' or graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] == 'V':
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'V'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'V'
            else:
                graph[event_ids_dict[triplet[0]], event_ids_dict[triplet[2]]] = 'C'
                graph[event_ids_dict[triplet[2]], event_ids_dict[triplet[0]]] = 'C'

    # returning the upper triangular matrix
    return numpy.triu(graph, k=1), event_ids_dict
