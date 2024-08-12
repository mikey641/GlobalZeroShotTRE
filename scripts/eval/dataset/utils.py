import math
import re


def count_stats_in_file(data):
    mentions = data["allMentions"]
    final_mentions = list()

    for mention in mentions:
        if mention['axisType'] == 'main':
            final_mentions.append(mention)

    num_mentions = len(final_mentions)
    num_tmp_pairs = len(data["allPairs"])
    num_equal_pairs = 0
    num_before_after_pairs = 0
    num_vague_pairs = 0

    all_pairs = data["allPairs"]
    for pair in all_pairs:
        if pair['_relation'].startswith('equal'):
            num_equal_pairs += 1
        elif pair['_relation'].startswith('before') or pair['_relation'].startswith('after'):
            num_before_after_pairs += 1
        elif pair['_relation'].startswith('uncertain'):
            num_vague_pairs += 1

    expected_pairs = (math.pow(num_mentions, 2) - num_mentions) / 2
    # print(f'Number of mentions={num_mentions}')
    # print(f'Number of relations={num_tmp_pairs} (expected={expected_pairs})')
    return final_mentions, num_tmp_pairs, num_equal_pairs, num_before_after_pairs, num_vague_pairs, expected_pairs


def get_annotations(pairs):
    tmp_lables = list()
    coref_lables = list()
    causal_lables = list()
    for pair in pairs:
        first_id = pair['_firstId']
        second_id = pair['_secondId']
        relation = pair['_relation']
        if '/' in relation:
            split_rel = relation.split('/')
            tmp_lables.append((first_id, second_id, split_rel[0]))
            if split_rel[0] == 'before' or split_rel[0] == 'after':
                causal_lables.append((first_id, second_id, split_rel[1]))
            elif split_rel[0] == 'equal':
                coref_lables.append((first_id, second_id, split_rel[1]))
            else:
                raise ValueError(f'Unknown relation {split_rel[0]}')
        else:
            tmp_lables.append((first_id, second_id, relation))

    tmp_lables.sort(key=lambda tup: (tup[0], tup[1]))
    coref_lables.sort(key=lambda tup: tup[0])
    causal_lables.sort(key=lambda tup: tup[0])
    return tmp_lables, coref_lables, causal_lables


def find_diffs(mentions, labs1, labs2):
    diffs = list()
    sames = list()
    most_unagreed = dict()
    for i in range(len(labs1)):
        ment1 = find_ment_by_id(mentions, labs1[i][0])
        text_m1 = f'{ment1["tokens"]}({ment1["m_id"]})'
        ment2 = find_ment_by_id(mentions, labs1[i][1])
        text_m2 = f'{ment2["tokens"]}({ment2["m_id"]})'
        if labs1[i] == labs2[i]:
            sames.append([text_m1, text_m2, labs1[i][2], labs2[i][2]])
        if labs1[i] != labs2[i]:
            diffs.append([text_m1, text_m2, labs1[i][2], labs2[i][2]])

            if text_m1 not in most_unagreed:
                most_unagreed[text_m1] = 0
            if text_m2 not in most_unagreed:
                most_unagreed[text_m2] = 0
            most_unagreed[text_m1] += 1
            most_unagreed[text_m2] += 1

    return diffs, sames, most_unagreed


def find_ment_by_id(mentions, m_id):
    for mention in mentions:
        if mention['m_id'] == m_id:
            return mention

    return None


def parse_DOT(dot_json):
    if 'strict graph' not in dot_json:
        print("Invalid DOT file!!!!!")
        return None

    edges = dot_json[dot_json.index('strict graph')+len('strict graph {'):].split(';')
    key_set = set()
    graph = [] # graph edge list
    duplicate = 0
    for edge_str in edges:
        rel_list = re.findall(r'rel=([a-zA-Z]+)', edge_str)

        if len(rel_list) < 1:
            break

        rel = rel_list[0].lower()

        if rel not in ['after', 'before', 'equal', 'vague']: #['after', 'before']:
            continue

        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split('--')) < 2:
            continue

        event_1 = event_pair.split(' -- ')[0].lower().strip()
        event_2 = event_pair.split(' -- ')[1].lower().strip()

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

        key = f"{event_1}||{rel}||{event_2}"
        if key in key_set:
            duplicate += 1
        else:
            graph.append((event_1, rel, event_2))
            key_set.add(key)
            # print(event_1, rel, event_2)
    #print(f"Num of duplication: {duplicate}")
    return graph, duplicate
