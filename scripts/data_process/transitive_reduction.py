import json

from scripts.data_process.convert_mydata_to_dot import read_files_from_folder, convert_doc_to_dot

import numpy as np

from scripts.utils.transitive_algos import transitive_reduction_with_relations, transitive_closure_with_relations, \
    json_to_numpy_graph


def remove_trans_pairs(reduction_relations, event_ids, content):
    new_pairs = []
    for pair in content['pairs']:
        if reduction_relations[event_ids[pair['_firstId']], event_ids[pair['_secondId']]] != '':
            new_pairs.append(pair)

    return new_pairs


def build_graphs(files_content):
    for filename, content in files_content.items():
        graph, event_ids = json_to_numpy_graph(content)
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


