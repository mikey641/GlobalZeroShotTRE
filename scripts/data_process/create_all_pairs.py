import json
import math
import os
import random

from tqdm import tqdm

from scripts.prompting_global.jup_utils import filter_non_events
from scripts.utils.io_utils import open_input_file


def reverse_label(label):
    if label == 'BEFORE':
        return 'AFTER'
    elif label == 'AFTER':
        return 'BEFORE'
    elif label == 'IS_INCLUDED':
        return 'INCLUDES'
    elif label == 'INCLUDES':
        return 'IS_INCLUDED'
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


def handle_chunks(all_pairs, all_mentions, tokens, max_pairs_in_chunk, test_out_folder):
    if len(all_pairs) > max_pairs_in_chunk:
        random.shuffle(all_pairs)
        chunks_size = math.ceil(len(all_pairs) / math.ceil(len(all_pairs) / max_pairs_in_chunk))
        chunk_pairs = [all_pairs[i:i + chunks_size] for i in range(0, len(all_pairs), chunks_size)]

        for j, chunk in enumerate(chunk_pairs):
            chunk_sorted = sorted(chunk, key=lambda x: x['index'])
            new_data = {'tokens': tokens, 'allMentions': ret_only_relevant_mentions(all_mentions, chunk_sorted),
                        'allPairs': chunk_sorted}
            with open(f'{test_out_folder}/{file_name}_chunk_{j}.json', 'w') as file:
                json.dump(new_data, file, indent=4)
    else:
        new_data = {'tokens': tokens, 'allMentions': all_mentions,
                    'allPairs': sorted(all_pairs, key=lambda x: x['index'])}
        with open(f'{test_out_folder}/{file1}', 'w') as file:
            json.dump(new_data, file, indent=4)


def ret_only_relevant_mentions(mentions, pairs):
    relevant_mentions = set()
    for pair in pairs:
        relevant_mentions.add(pair['_firstId'])
        relevant_mentions.add(pair['_secondId'])
    return [mention for mention in mentions if mention['m_id'] in relevant_mentions]


if __name__ == "__main__":
    _max_pairs_in_chunk = 100
    _test_folder = 'data/TimeBank-Dense/testOnlyProb'
    _test_out_folder = 'data/TimeBank-Dense/testOnlyProb_allpairs'
    for i, file1 in enumerate(tqdm(os.listdir(_test_folder))):
        file_name, file_extension = os.path.splitext(file1)

        _data = open_input_file(f'{_test_folder}/{file1}')
        _tokens, _mentions, _all_new_pairs, _pairs_with_rel = get_data(_data)

        assert len(_all_new_pairs) == (len(_mentions) * (len(_mentions) - 1)) / 2
        assert _pairs_with_rel == len(_data['allPairs'])

        handle_chunks(_all_new_pairs, _mentions, _tokens, _max_pairs_in_chunk, _test_out_folder)
        with open(f'{_test_out_folder}/{file1}', 'w') as file:
            json.dump({'tokens': _tokens, 'allMentions': _mentions, 'allPairs': _all_new_pairs}, file, indent=4)
