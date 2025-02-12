import json
import os
from collections import Counter


def main(narrative_folder, narrative_folder_no_overlap):
    # read all files in folder
    sanity = list()
    mentions_classes = list()
    for nar_file in os.listdir(narrative_folder):
        with open(f'{narrative_folder}/{nar_file}', mode='r') as file:
            overlap_nodes = list()
            overlap_pairs = list()
            data = json.load(file)
            all_pairs = data['allPairs']
            all_mentions = data['allMentions']
            mentions_classes.extend([ment['event_class'] for ment in all_mentions])
            for pair in all_pairs:
                rel = pair['_relation']
                if rel == 'overlap':
                    overlap_nodes.append(pair['_firstId'])
                    overlap_nodes.append(pair['_secondId'])
                    overlap_pairs.append((pair['_firstId'], pair['_secondId']))

            overlap_nodes = Counter(overlap_nodes)
            sorted_counter = overlap_nodes.most_common()
            removed_nodes = list()
            for node, count in sorted_counter:
                if len(overlap_pairs) != 0:
                    removed_nodes.append(node)
                    for i in range(len(overlap_pairs) - 1, -1, -1):
                        ov_pair = overlap_pairs[i]
                        if node in ov_pair:
                            overlap_pairs.remove(ov_pair)

            new_mentions = [ment for ment in all_mentions if ment['m_id'] not in removed_nodes]
            new_pairs = [pair for pair in all_pairs if pair['_firstId'] not in removed_nodes and pair['_secondId'] not in removed_nodes]
            new_data = {'tokens': data['tokens'], 'allMentions': new_mentions, 'allPairs': new_pairs}
            with open(f'{narrative_folder_no_overlap}/{nar_file}', mode='w') as new_file:
                json.dump(new_data, new_file, indent=4)

            sanity.extend([pair['_relation'] for pair in new_pairs])

    print(Counter(mentions_classes))
    print(Counter(sanity))


if __name__ == '__main__':
    _narrative_folder = "data/NarrativeTime_A2/converted"
    _narrative_folder_no_overlap = "data/NarrativeTime_A2/converted_no_overlap"
    main(_narrative_folder, _narrative_folder_no_overlap)
