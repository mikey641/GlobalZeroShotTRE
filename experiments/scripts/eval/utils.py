import math


def count_stats_in_file(data):
    mentions = data["allMentions"]
    final_mentions = list()

    for mention in mentions:
        if mention['axisType'] == 'main':
            final_mentions.append(mention)

    num_mentions = len(final_mentions)
    num_pairs = len(data["allPairs"])
    print(f'Number of mentions={num_mentions}')
    print(f'Number of relations={num_pairs} (expected={(math.pow(num_mentions, 2) - num_mentions)/2})')


def get_annotations(data):
    tmp_lables = list()
    coref_lables = list()
    causal_lables = list()
    for pair in data['allPairs']:
        relation = pair['_relation']
        if '/' in relation:
            split_rel = relation.split('/')
            tmp_lables.append(split_rel[0])
            if split_rel[0] == 'before' or split_rel[0] == 'after':
                causal_lables.append(split_rel[1])
            elif split_rel[0] == 'equal':
                coref_lables.append(split_rel[1])
            else:
                print("Not suppose to be here")
        else:
            tmp_lables.append(relation)

    return tmp_lables, coref_lables, causal_lables


def find_diffs(data, lab1, lab2):
    diffs = list()
    most_unagreed = dict()
    mentions = data['allMentions']
    pairs = data['allPairs']
    for i in range(len(lab1)):
        if lab1[i] != lab2[i]:
            ment1 = find_ment_by_id(mentions, pairs[i]['_firstId'])
            text_m1 = f'{ment1["tokens"]}({pairs[i]["_firstId"]})'
            ment2 = find_ment_by_id(mentions, pairs[i]['_secondId'])
            text_m2 = f'{ment2["tokens"]}({pairs[i]["_secondId"]})'
            diffs.append([text_m1, text_m2, lab1[i], lab2[i]])

            if text_m1 not in most_unagreed:
                most_unagreed[text_m1] = 0
            if text_m2 not in most_unagreed:
                most_unagreed[text_m2] = 0
            most_unagreed[text_m1] += 1
            most_unagreed[text_m2] += 1

    return diffs, most_unagreed


def find_ment_by_id(mentions, m_id):
    for mention in mentions:
        if mention['m_id'] == m_id:
            return mention

    return None
