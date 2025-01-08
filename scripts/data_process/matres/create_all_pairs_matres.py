import json
import os

from tqdm import tqdm

from scripts.prompting.jup_utils import open_input_file, filter_non_events


def reverse_label(label):
    if label == 'BEFORE':
        return 'AFTER'
    if label == 'AFTER':
        return 'BEFORE'
    return label


def get_data(data):
    uniqe_pairs_ids = set()
    all_new_pairs = []
    all_mentions = filter_non_events(data['allMentions'])
    all_pairs = data['allPairs']
    all_pairs_ids = {(pair['_firstId'], pair['_secondId']): pair for pair in all_pairs}

    pairs_with_rel = 0
    for ment1 in all_mentions:
        for ment2 in all_mentions:
            if ment1['m_id'] != ment2['m_id']:
                if (ment1['m_id'], ment2['m_id']) not in uniqe_pairs_ids and (ment2['m_id'], ment1['m_id']) not in uniqe_pairs_ids:
                    uniqe_pairs_ids.add((ment1['m_id'], ment2['m_id']))
                    uniqe_pairs_ids.add((ment2['m_id'], ment1['m_id']))

                    first = ment1
                    second = ment2
                    if ment1['tokens_ids'][0] > ment2['tokens_ids'][0]:
                        first = ment2
                        second = ment1

                    relation = 'NA'
                    if (first['m_id'], second['m_id']) in all_pairs_ids:
                        relation = all_pairs_ids[(first['m_id'], second['m_id'])]['_relation']
                        pairs_with_rel += 1
                    elif (second['m_id'], first['m_id']) in all_pairs_ids:
                        relation = reverse_label(all_pairs_ids[(second['m_id'], first['m_id'])]['_relation'])
                        pairs_with_rel += 1
                    all_new_pairs.append({'_firstId': first['m_id'],
                                      '_secondId': second['m_id'],
                                      'firstEventText': first['tokens'],
                                      'secondEventText': second['tokens'],
                                      'index': first['tokens_ids'][0],
                                      '_relation': relation})

    return data['tokens'], all_mentions, sorted(all_new_pairs, key=lambda x: x['index']), pairs_with_rel


def ret_only_relevant_mentions(mentions, pairs):
    relevant_mentions = set()
    for pair in pairs:
        relevant_mentions.add(pair['_firstId'])
        relevant_mentions.add(pair['_secondId'])
    return [mention for mention in mentions if mention['m_id'] in relevant_mentions]


if __name__ == "__main__":
    _max_pairs_in_chunk = 120
    _test_folder = 'data/MATRES/in_my_format/test'
    _test_out_folder = 'data/MATRES/in_my_format_all_pairs_120/test'
    for i, file1 in enumerate(tqdm(os.listdir(_test_folder))):
        file_name, file_extension = os.path.splitext(file1)

        _data = open_input_file(f'{_test_folder}/{file1}')
        _tokens, _mentions, _all_new_pairs, _pairs_with_rel = get_data(_data)

        assert len(_all_new_pairs) == (len(_mentions) * (len(_mentions) - 1)) / 2
        assert _pairs_with_rel == len(_data['allPairs'])

        if len(_all_new_pairs) > _max_pairs_in_chunk:
            chunk_pairs = [_all_new_pairs[i:i + _max_pairs_in_chunk] for i in range(0, len(_all_new_pairs), _max_pairs_in_chunk)]
            for j, chunk in enumerate(chunk_pairs):

                new_data = {'tokens': _tokens, 'allMentions': ret_only_relevant_mentions(_mentions, chunk), 'allPairs': chunk}
                with open(f'{_test_out_folder}/{file_name}_chunk_{j}.json', 'w') as file:
                    json.dump(new_data, file, indent=4)
        else:
            new_data = {'tokens': _tokens, 'allMentions': _mentions, 'allPairs': _all_new_pairs}
            with open(f'{_test_out_folder}/{file1}', 'w') as file:
                json.dump(new_data, file, indent=4)
