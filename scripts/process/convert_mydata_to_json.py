import json

from scripts.process.convert_mydata_to_dot import read_files_from_folder


def convert_doc_to_json(content):
    ret_json = dict()
    ret_json['document'] = " ".join(content['tokens'])
    all_ment = content['mentions']
    pairs_list = list()
    for pair in content['pairs']:
        mention1 = all_ment[pair['_firstId']]['tokens'] + '(' + str(pair['_firstId']) + ')'
        mention2 = all_ment[pair['_secondId']]['tokens'] + '(' + str(pair['_secondId']) + ')'

        relation = 'vague' if pair['_relation'] == 'uncertain' else pair['_relation']
        pair = {'e1': mention1, 'e2': mention2, 'relation': relation}
        pairs_list.append(pair)
    ret_json['target'] = pairs_list
    return ret_json


if __name__ == '__main__':
    _folder_path = 'data/EventFullTrainExports'
    _output_path = 'data/tmp/EventFull_notDot.json'
    _files_content = read_files_from_folder(_folder_path)

    _out_json = dict()
    for _filename, _content in _files_content.items():
        _out_json[_filename] = convert_doc_to_json(_content)
        # print(f"Filename: {_filename}\nContent:\n{_content}\n")

    with open(_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(_out_json, json_file, ensure_ascii=False, indent=4)
