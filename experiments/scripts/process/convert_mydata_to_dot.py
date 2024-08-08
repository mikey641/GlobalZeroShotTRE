import json
import os
from collections import OrderedDict


def read_mentions(mentions):
    ment_dict = dict()
    for ment in mentions:
        ment_dict[ment['m_id']] = ment
    return ment_dict


def read_files_from_folder(folder_path):
    files_content = OrderedDict()
    files_list = list()
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                files_list.append(file_path)

        files_list.sort(key=lambda name: int(os.path.basename(name).split("_")[0]))
        for file_path in files_list:
            with open(file_path, 'r') as file:
                data = json.load(file)
                mentions = read_mentions(data['allMentions'])
                pairs = data['allPairs']
                files_content[os.path.basename(file.name)] = {"tokens": data['tokens'], "mentions": mentions, "pairs": pairs}
    except Exception as e:
        print(f"An error occurred: {e}")
    return files_content


def convert_doc_to_dot(content):
    ret_json = dict()
    ret_json['document'] = " ".join(content['tokens'])
    all_ment = content['mentions']
    pair_str_list = list()
    for pair in content['pairs']:
        mention1 = all_ment[pair['_firstId']]['tokens']
        mention2 = all_ment[pair['_secondId']]['tokens']
        pair_str_list.append('"' + mention1 + '"'' -- ''"' + mention2 + '" [rel=' + pair['_relation'] + '];')
    ret_json['target'] = "strict graph {\n\n" + "\n".join(pair_str_list) + "\n\n}"
    return ret_json


if __name__ == '__main__':
    _folder_path = 'data/my_data/EventFullTrainExports'
    _files_content = read_files_from_folder(_folder_path)

    _out_json = dict()
    for _filename, _content in _files_content.items():
        _out_json[_filename] = convert_doc_to_dot(_content)
        # print(f"Filename: {_filename}\nContent:\n{_content}\n")

    with open("data/tmp/my_data_dot.json", 'w', encoding='utf-8') as json_file:
        json.dump(_out_json, json_file, ensure_ascii=False, indent=4)
