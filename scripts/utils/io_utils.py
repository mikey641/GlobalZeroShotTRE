import os
import re
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import json

from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, DataType


@dataclass(frozen=True)
class Event_Rel:
    docid: str
    label: str      # label is a word, BEFORE, AFTER...
    source: str
    target: str
    token: list
    event_ix: list
    verbs: list
    lemma: list
    part_of_speech: list
    position: list
    length: int
    sentdiff: int


def load_xml(xml_element):
    xml_element = xml_element
    label = xml_element.attrib['LABEL']
    sentdiff = int(xml_element.attrib['SENTDIFF'])
    docid = xml_element.attrib['DOCID']
    source = xml_element.attrib['SOURCE']
    target = xml_element.attrib['TARGET']
    data = xml_element.text.strip().split()
    token = []
    lemma = []
    part_of_speech = []
    position = []
    length = len(data)
    event_ix = []
    verbs = []

    for i,d in enumerate(data):
        tmp = d.split('///')
        part_of_speech.append(tmp[-2])
        position.append(tmp[-1])
        if tmp[-1] == 'E1':
            event_ix.append(i)
            verbs.append(tmp[0])
        elif tmp[-1] == 'E2':
            event_ix.append(i)
            verbs.append(tmp[0])
        token.append(tmp[0])
        lemma.append(tmp[1])

    if len(event_ix) < 2:
        print()

    return Event_Rel(docid=docid, label=label, source=source,
                    target=target, token=token, lemma=lemma,
                    part_of_speech=part_of_speech, position=position,
                    length=length, event_ix=event_ix, verbs=verbs, sentdiff=sentdiff)


def read_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    orig_ins_list = []

    for e in root:
        ins = load_xml(e)
        orig_ins_list.append(ins)

    docs_dict = dict()
    for ins in orig_ins_list:
        if ins.docid not in docs_dict:
            docs_dict[ins.docid] = dict()
            docs_dict[ins.docid]['rels'] = list()
            docs_dict[ins.docid]['events'] = set()

        docs_dict[ins.docid]['rels'].append(ins.label)
        docs_dict[ins.docid]['events'].add(ins.source)
        docs_dict[ins.docid]['events'].add(ins.target)

    return docs_dict, orig_ins_list


def parse_DOT(dot_json, labels):
    if 'Strict graph' in dot_json:
        dot_json = dot_json.replace('Strict graph', 'strict graph')

    if 'strict graph' not in dot_json and 'Strict graph' not in dot_json:
        print("Invalid DOT file!!!!!")
        return None

    edges = dot_json[dot_json.index('strict graph')+len('strict graph {'):dot_json.rfind('```')].split(';')
    graph = [] # graph edge list
    for edge_str in edges:
        rel_list = re.findall(r'rel\s?=\s?"?([a-zA-Z_]+)"?', edge_str)

        if len(rel_list) < 1:
            break

        rel = rel_list[0].lower()
        if rel.endswith('s') and rel != 'simultaneous':
            rel = rel[:-1]

        if rel not in ['after', 'before', 'equal', 'vague', 'include', 'included', 'is_included', 'same', 'same_time', 'simultaneous', 'precede', 'during']: #['after', 'before']:
            continue

        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split('--')) < 2:
            continue

        event_1 = event_pair.split('--')[0].lower().strip()
        event_2 = event_pair.split('--')[1].lower().strip()

        if event_1[0] == ' ':
            event_1 = event_1[1:]

        event_1 = re.sub(r'\"', '', event_1)
        event_2 = re.sub(r'\"', '', event_2)

        if len(event_1) == 0 or len(event_2) == 0:
            continue
        if event_1 == " " or event_2 == " ":
            continue
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        if event_2[0] == ' ':
            event_2 = event_2[1:]
        if event_1[-1] == ' ':
            event_1 = event_1[:-1]
        if event_2[-1] == ' ':
            event_2 = event_2[:-1]

        graph.append((event_1, labels.adjust_label(rel.upper()), event_2))
        # print(event_1, rel, event_2)
    #print(f"Num of duplication: {duplicate}")
    return graph


def read_pred_dot_file_matres(predictions, labels):
    final_preds = dict()
    for pred_file in predictions.keys():
        final_preds[pred_file] = predictions[pred_file]

    pred_as_dict = dict()
    pred_triplets = dict()
    for file in final_preds.keys():
        # if 'bbc_20130322_1353' in file:
        #     print()
        predicted_graph = parse_DOT(final_preds[file]['target'], labels)
        if predicted_graph is None:
            print(f"Error: {file} has no predictions.")
            continue

        file_name_split = file.split('_')
        if 'chunk' in file_name_split:
            file = '_'.join(file_name_split[:-2])
        else:
            file = file.removesuffix(".json")

        if file not in pred_triplets:
            pred_triplets[file] = list()

        pred_triplets[file].extend(predicted_graph)

        for line in predicted_graph:
            source = re.search(r'\((\d+)\)', line[0])
            target = re.search(r'\((\d+)\)', line[2])
            source_id = 'NA'
            target_id = 'NA'
            relation = line[1]
            if source and target:
                source_id = source.group(1)
                target_id = target.group(1)
            else:
                print("Source/Target not found.")

            pred_as_dict[f'{file}#{source_id}#{target_id}'] = labels[labels.adjust_label(relation.upper())]

    return pred_as_dict, pred_triplets


def read_pred_dot_file(pred_file_path, test_docs_dict, data_type: DataType):
    with open(pred_file_path) as f:
        predictions = json.load(f)

    labels = data_type.get_label_set()
    if data_type.get_name() == MATRES_DATASET_NAME:
        return read_pred_dot_file_matres(predictions, labels)

    final_preds = dict()
    for pred_file in predictions.keys():
        fixed_file = pred_file
        file_name_split = pred_file.split('_')
        if 'chunk' in file_name_split:
            fixed_file = '_'.join(file_name_split[:-2]) + ".json"

        if fixed_file in test_docs_dict:
            if fixed_file not in final_preds:
                final_preds[fixed_file] = [predictions[pred_file]]
            else:
                final_preds[fixed_file].append(predictions[pred_file])

    if len(test_docs_dict) != len(final_preds):
        print("Error: Number of files in gold and prediction are different")

    pred_as_dict = dict()
    pred_triplets = dict()
    for file in final_preds.keys():
        all_preds = []
        for pred in final_preds[file]:
            predicted_graph = parse_DOT(pred['target'], data_type.get_label_set())
            if predicted_graph is None:
                print(f"Error: {file} has no predictions.")
                continue

            all_preds.extend(predicted_graph)

        pred_triplets[file] = all_preds

        for line in all_preds:
            source = re.search(r'\((\d+)\)', line[0])
            target = re.search(r'\((\d+)\)', line[2])
            source_id = 'NA'
            target_id = 'NA'
            relation = data_type.get_label_set().adjust_label(line[1].upper())
            if source and target:
                source_id = source.group(1)
                target_id = target.group(1)
            else:
                print("Source/Target not found.")
                continue

            pred_as_dict[f'{file}#{source_id}#{target_id}'] = labels[labels.adjust_label(relation)]

    return pred_as_dict, pred_triplets


def open_input_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def load_json_lines(jsonl_file):
    """
    Load a JSON Lines file into a list of dictionaries.
    Each line in the file should be a valid JSON object.
    """
    data = []
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
    return data
