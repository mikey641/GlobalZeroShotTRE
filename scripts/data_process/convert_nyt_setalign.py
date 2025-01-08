import json

from scripts.eval.dataset.utils import parse_DOT


def convert_doc_to_dot(content):
    ret_json = dict()
    ret_json['document'] = " ".join(content['tokens'])
    all_ment = content['mentions']
    pair_str_list = list()
    for pair in content['pairs']:
        mention1 = all_ment[pair['_firstId']]['tokens'] + '(' + str(pair['_firstId']) + ')'
        mention2 = all_ment[pair['_secondId']]['tokens'] + '(' + str(pair['_secondId']) + ')'

        relation = 'vague' if pair['_relation'] == 'uncertain' else pair['_relation']
        pair_str_list.append('"' + mention1 + '"'' -- ''"' + mention2 + '" [rel=' + relation.upper() + '];')
    ret_json['target'] = "strict graph {\n\n" + "\n".join(pair_str_list) + "\n\n}"
    return ret_json


def convert(golds):
    golds_convert = dict()
    for file in golds.keys():
        print(f'-------------- File: {file} ------------------')
        gold_graph = parse_DOT(golds[file]['target'])
        if gold_graph is None:
            print(f'Error in parsing {file}')
            continue

        golds_convert[file] = dict()
        golds_convert[file]['document'] = golds[file]['document']

        gold_graph_converted = list()
        for triplet in gold_graph:
            if triplet[1] == 'simultaneous':
                gold_graph_converted.append((triplet[0], 'EQUAL', triplet[2]))
            elif triplet[1] == 'includes':
                gold_graph_converted.append((triplet[0], 'BEFORE', triplet[2]))
            elif triplet[1] == 'is_included':
                gold_graph_converted.append((triplet[0], 'AFTER', triplet[2]))
            else:
                gold_graph_converted.append((triplet[0], triplet[1].upper(), triplet[2]))

        convert_doc_to_dot(gold_graph_converted)

        with open(out_file, 'w') as f:
            json.dump(golds, f, indent=4)


if __name__ == "__main__":
    in_file = "data/NYT_SetAlign/NYT_des_human.json"
    out_file = "data/NYT_SetAlign/NYT_des_human_bf_af_eq.json"

    with open(in_file) as f:
        _golds = json.load(f)

    convert(_golds)
