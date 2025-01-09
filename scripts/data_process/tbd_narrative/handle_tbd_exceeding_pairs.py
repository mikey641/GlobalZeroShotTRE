import json
import math
import os

from tqdm import tqdm

from scripts.data_process.matres.create_all_pairs_matres import get_data, ret_only_relevant_mentions
from scripts.prompting.jup_utils import open_input_file

if __name__ == "__main__":
    _max_pairs_in_chunk = 200
    _test_folder = 'data/TimeBank-Dense/test_converted'
    _test_out_folder = 'data/TimeBank-Dense/test_converted_managed_size'
    for i, file1 in enumerate(tqdm(os.listdir(_test_folder))):
        file_name, file_extension = os.path.splitext(file1)

        _data = open_input_file(f'{_test_folder}/{file1}')
        _tokens = _data['tokens']
        _all_mentions = _data['allMentions']
        _all_pairs = _data['allPairs']

        _all_ment_ids = {m['m_id']: m for m in _all_mentions}

        for pair in _all_pairs:
            pair['index'] = _all_ment_ids[pair['_firstId']]['tokens_ids'][0]

        all_sorted_pairs = sorted(_all_pairs, key=lambda x: x['index'])

        if len(all_sorted_pairs) > _max_pairs_in_chunk:
            chunks_size = math.ceil(len(all_sorted_pairs) / math.ceil(len(all_sorted_pairs) / _max_pairs_in_chunk))
            chunk_pairs = [all_sorted_pairs[i:i + chunks_size] for i in range(0, len(all_sorted_pairs), chunks_size)]
            for j, chunk in enumerate(chunk_pairs):
                new_data = {'tokens': _tokens, 'allMentions': ret_only_relevant_mentions(_all_mentions, chunk), 'allPairs': chunk}
                with open(f'{_test_out_folder}/{file_name}_chunk_{j}.json', 'w') as _file:
                    json.dump(new_data, _file, indent=4)
        else:
            new_data = {'tokens': _tokens, 'allMentions': _all_mentions, 'allPairs': all_sorted_pairs}
            with open(f'{_test_out_folder}/{file1}', 'w') as _file:
                json.dump(new_data, _file, indent=4)
