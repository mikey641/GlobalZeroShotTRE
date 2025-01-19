import json
import os
import random
from collections import Counter


def main(narrative_folder, narrative_folder_no_overlap):
    # read all files in folder
    sanity = list()
    for nar_file in os.listdir(narrative_folder):
        with open(f'{narrative_folder}/{nar_file}', mode='r') as file:
            data = json.load(file)
            all_pairs = data['allPairs']
            all_mentions = data['allMentions']
            all_ids = [ment['m_id'] for ment in all_mentions]
            ment_sample_ids = random.sample(all_ids, 18)
            all_new_ment = [ment for ment in all_mentions if ment['m_id'] in ment_sample_ids]

            new_pairs = [pair for pair in all_pairs if pair['_firstId'] in ment_sample_ids and pair['_secondId'] in ment_sample_ids]
            new_data = {'tokens': data['tokens'], 'allMentions': all_new_ment, 'allPairs': new_pairs}
            with open(f'{narrative_folder_no_overlap}/{nar_file}', mode='w') as new_file:
                json.dump(new_data, new_file, indent=4)

            sanity.extend([pair['_relation'] for pair in new_pairs])

    print(Counter(sanity))


if __name__ == '__main__':
    _narrative_folder = "data/NarrativeTime/converted_no_overlap/test"
    _narrative_folder_no_overlap = "data/NarrativeTime/converted_no_overlap/test_18ment"
    main(_narrative_folder, _narrative_folder_no_overlap)
