import json

from scripts.process.convert_mydata_to_dot import read_files_from_folder, convert_doc_to_dot

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


def remove_trans_pairs(reduction_relations, event_ids, content):
    new_pairs = []
    for pair in content['pairs']:
        if reduction_relations[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] != '':
            new_pairs.append(pair)

    return new_pairs


def build_graphs(files_content):
    for filename, content in files_content.items():
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

        reduction_relations, total_reduced = transitive_reduction_with_relations(graph)
        back_to_transitive = transitive_closure_with_relations(reduction_relations)
        assert np.array_equal(graph, back_to_transitive), "Sanity Check Failed! Transitive closure and reduction are not consistent"
        new_pairs = remove_trans_pairs(reduction_relations, event_ids, content)
        assert len(new_pairs) == (len(content['pairs']) - total_reduced), "Sanity Check Failed! Number of pairs is not consistent"
        content['pairs'] = new_pairs


if __name__ == "__main__":
    _folder_path = 'data/EventFullTrainExports/all'
    _output_path = 'data/DOT_format/trans_reduced/EventFull_all_dot.json'

    _files_content = read_files_from_folder(_folder_path)
    build_graphs(_files_content)

    _out_json = dict()
    for _filename, _content in _files_content.items():
        _out_json[_filename] = convert_doc_to_dot(_content)
        # print(f"Filename: {_filename}\nContent:\n{_content}\n")

    with open(_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(_out_json, json_file, ensure_ascii=False, indent=4)

    print('Done!')


