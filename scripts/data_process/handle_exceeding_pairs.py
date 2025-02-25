import json
import math
import os
import random

from tqdm import tqdm

from scripts.data_process.create_all_pairs import ret_only_relevant_mentions, handle_chunks
from scripts.prompting_global.jup_utils import open_input_file


def reduce_pairs_random(test_folder, max_pairs_in_chunk, test_out_folder):
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        file_name, file_extension = os.path.splitext(file1)

        data = open_input_file(f'{test_folder}/{file1}')
        tokens = data['tokens']
        all_mentions = data['allMentions']
        all_pairs = data['allPairs']

        all_ment_ids = {m['m_id']: m for m in all_mentions}

        for pair in all_pairs:
            pair['index'] = all_ment_ids[pair['_firstId']]['tokens_ids'][0]

        if len(all_pairs) > max_pairs_in_chunk:
            reduced_ment = [ment for ment in all_mentions if ment['event_class'] == 'OCCURRENCE']
            reduced_ment_ids = {m['m_id']: m for m in reduced_ment}
            filtered_pairs = [pair for pair in all_pairs if pair['_firstId'] in reduced_ment_ids and pair['_secondId'] in reduced_ment_ids]
            final_ment = ret_only_relevant_mentions(all_mentions, filtered_pairs)

            new_data = {'tokens': tokens, 'allMentions': final_ment, 'allPairs': filtered_pairs}
            with open(f'{test_out_folder}/{file_name}.json', 'w') as file:
                json.dump(new_data, file, indent=4)
        else:
            new_data = {'tokens': tokens, 'allMentions': all_mentions, 'allPairs': sorted(all_pairs, key=lambda x: x['index'])}
            with open(f'{test_out_folder}/{file1}', 'w') as file:
                json.dump(new_data, file, indent=4)



def chunk_by_edges(test_folder, max_pairs_in_chunk, test_out_folder):
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        file_name, file_extension = os.path.splitext(file1)

        data = open_input_file(f'{test_folder}/{file1}')
        tokens = data['tokens']
        all_mentions = data['allMentions']
        all_pairs = data['allPairs']

        all_ment_ids = {m['m_id']: m for m in all_mentions}

        for pair in all_pairs:
            pair['index'] = all_ment_ids[pair['_firstId']]['tokens_ids'][0]

        handle_chunks(all_pairs, all_mentions, tokens, max_pairs_in_chunk, test_out_folder)


if __name__ == "__main__":
    _max_pairs_in_chunk = 200
    _test_folder = 'data/TimeBank-Dense/test_converted_allpairs'
    _test_out_folder = 'data/TimeBank-Dense/test_converted_allpairs_managed_size'
    # chunk_by_node_size(_test_folder, _max_pairs_in_chunk, _test_out_folder)
    reduce_pairs_random(_test_folder, _max_pairs_in_chunk, _test_out_folder)
    print("Done")

