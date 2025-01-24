import json
import os

if __name__ == '__main__':
    _folder = "data/NarrativeTime/converted_no_overlap/test_18ment"
    _relation_distribution = dict()
    _total_pairs = 0
    _total_events = 0
    _max_doc_mention = 0
    _max_doc_pairs = 0
    _max_doc_name = ""
    for file in os.listdir(_folder):
        if file.endswith('.json'):
            with open(f'{_folder}/{file}') as f:
                data = json.load(f)
                mentions = data['allMentions']
                pairs_ = data['allPairs']
                _total_pairs += len(pairs_)
                _total_events += len(mentions)

                if _max_doc_mention < len(mentions):
                    _max_doc_mention = len(mentions)
                    _max_doc_pairs = len(pairs_)
                    _max_doc_name = file

                for pair in pairs_:
                    relation = pair['_relation']
                    if relation in _relation_distribution:
                        _relation_distribution[relation] += 1
                    else:
                        _relation_distribution[relation] = 1

    print(_relation_distribution)
    print(f"Total pairs: {_total_pairs}")
    print(f"Total events: {_total_events}")
    print(f"Max doc mention: {_max_doc_mention}")
    print(f"Max doc pairs: {_max_doc_pairs}")
    print(f"Max doc name: {_max_doc_name}")
