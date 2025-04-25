import json
import os

from scripts.utils.io_utils import open_input_file
from scripts.utils.omni_format_utils import filter_non_events


def prompt_zero_shot(source, target):
    return f"""Given the document D and a list of temporal relations [before, after, vague, equal] and event triggers that are labeled as [EVENT][/EVENT]. what is the temporal relation between <EVENT {source['m_id']}>{source['tokens']}</EVENT> and <EVENT {target['m_id']}>{target['tokens']}</EVENT>? Answer vague if unsure. Respond only with the answer (one of: before, after, vague, equal.)"""


def prompt_cot(source, target):
    return f"""Given the document D, are <EVENT {source['m_id']}>{source['tokens']}</EVENT> and <EVENT {target['m_id']}>{target['tokens']}</EVENT> referring to the same event? Answer yes or no."""


def prompt_first_timeline_then_rel(source, target):
    return f"""Given the document D and a list of temporal relations [before, after, vague, equal], provide a brief explanation of the timeline between the source <EVENT {source['m_id']}>{source['tokens']}</EVENT> and target <EVENT {target['m_id']}>{target['tokens']}</EVENT> events. Conclude with your final decision based on your timeline interpretation in the format: Relation=Your Response."""


def mark_events_in_text(tokens, all_mentions):
    for mention in all_mentions:
        tok_first_id = mention['tokens_ids'][0]
        tok_last_id = mention['tokens_ids'][-1]
        tokens[tok_first_id] = f'<EVENT {mention["m_id"]}>{tokens[tok_first_id]}'
        tokens[tok_last_id] = f'{tokens[tok_last_id]}</EVENT>'
    return " ".join(tokens)


def get_input_text(data):
    if data is not None:
        tokens = data['tokens']
        all_mentions = filter_non_events(data['allMentions'])
        all_mentions.sort(key=lambda x: x['tokens_ids'][0])
        text = mark_events_in_text(tokens, all_mentions)
        text = f'Input document D = {text}'
        return text, all_mentions


def prepare_instructions(test_folder, instructions_func):
    all_prompts = list()
    for file1 in os.listdir(test_folder):
        print(f'generating file: {file1}')
        data = open_input_file(f'{test_folder}/{file1}')
        intput_text, mentions = get_input_text(data)
        all_pairs = data['allPairs']
        all_ments_by_id = {m['m_id']: m for m in mentions}
        for pair in all_pairs:
            instructions = instructions_func(all_ments_by_id[pair['_firstId']], all_ments_by_id[pair['_secondId']])

            all_prompts.append({
                'file': file1,
                'source': pair['_firstId'],
                'source_text': all_ments_by_id[pair['_firstId']]['tokens'],
                'target': pair['_secondId'],
                'target_text': all_ments_by_id[pair['_secondId']]['tokens'],
                'instruct': f'{intput_text}\n{instructions}',
                'gold_label': pair['_relation']})

    return all_prompts


if __name__ == "__main__":
    _instructions = prompt_zero_shot

    # _test_folder = 'data/OmniTemp/test'
    # _output_file = f'data/my_data/zero_shot/eventfull_cot_prompts.jsonl'

    _test_folder = 'data/MATRES/in_my_format/test'
    _output_file = f'data/my_data/zero_shot/matres_zero_shot_prompts.jsonl'

    # _test_folder = 'data/TimeBank-Dense/test_converted'
    # _output_file = f'data/my_data/zero_shot/tbd_remove_large_docs_cot_prompts.jsonl'

    # _test_folder = 'data/TimeBank-Dense/test_converted'
    # _output_file = f'data/my_data/zero_shot/tbd_cot_prompts.jsonl'

    examples = prepare_instructions(_test_folder, _instructions)
    # Write the list as json list file
    with open(_output_file, 'w') as _file:
        json.dump(examples, _file, indent=4, ensure_ascii=False)

    print(f"total examples: {len(examples)}")
