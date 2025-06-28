import json
import os

if __name__ == '__main__':
    _folder = "data/NarrativeTime_A1/converted_no_overlap/test_18ment"
    _relation_distribution = dict()
    _total_pairs = 0
    _total_events = 0
    _max_doc_mention = 0
    _max_doc_pairs = 0
    _max_doc_name = ""
    _total_tokens = 0
    _max_tokens = 0
    _num_docs = 0
    for file in os.listdir(_folder):
        if file.endswith('.json'):
            with open(f'{_folder}/{file}') as f:
                data = json.load(f)
                tokens = data['tokens']
                mentions = data['allMentions']
                mentions = [m for m in mentions if m['axisType'] == 'main']
                pairs_ = data['allPairs']
                _total_pairs += len(pairs_)
                _total_events += len(mentions)
                _total_tokens += len(tokens)
                _num_docs += 1

                if _max_doc_mention < len(mentions):
                    _max_doc_mention = len(mentions)
                    _max_doc_pairs = len(pairs_)
                    _max_doc_name = file

                if _max_tokens < len(tokens):
                    _max_tokens = len(tokens)

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
    print(f"Average tokens: {_total_tokens / _num_docs}")
    print(f"Max tokens doc: {_max_tokens}")
