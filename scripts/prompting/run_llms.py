import json
import os

import google.generativeai as genai
from openai import OpenAI

from scripts.eval.utils import find_ment_by_id
from scripts.prompting.prompts import task_description_v2


def run_gpt4_turbo(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def run_gpt4(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def run_gpt3_5(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def run_gemini_pro(_prompt):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)
    response = model.generate_content(_prompt)
    return response.text


def filter_non_events(events):
    return [e for e in events if e['axisType'] == 'main']


def mark_events_in_text(tokens, all_mentions):
    for mention in all_mentions:
        tok_first_id = mention['tokens_ids'][0]
        tok_last_id = mention['tokens_ids'][-1]
        tokens[tok_first_id] = f'<{tokens[tok_first_id]}'
        tokens[tok_last_id] = f'{tokens[tok_last_id]}({mention["m_id"]})>'
    return " ".join(tokens)


def get_example_matrix(pairs, all_ment_ids):
    example_matrix = [[0 for _ in range(len(all_ment_ids))] for _ in range(len(all_ment_ids))]
    for pair in pairs:
        first_id = pair['_firstId']
        second_id = pair['_secondId']
        relation = pair['_relation']
        if '/' in relation:
            split_rel = relation.split('/')
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = split_rel[0]
        else:
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = relation

    # pretty print matrix
    print("Expected matrix:")
    for row in example_matrix:
        print(row)

    return example_matrix


def get_prompt_pairs(mentions, gold_pairs):
    prompt_pairs = list()
    example_with_rels = list()
    for pair in gold_pairs:
        first_ment = find_ment_by_id(mentions, pair['_firstId'])
        second_ment = find_ment_by_id(mentions, pair['_secondId'])
        first_pair = f'{first_ment["tokens"]}({first_ment["m_id"]})'
        second_pair = f'{second_ment["tokens"]}({second_ment["m_id"]})'
        prompt_pairs.append(f'{first_pair}/{second_pair}')
        example_with_rels.append(f'{first_pair}/{second_pair}={pair["_relation"]}')
    return prompt_pairs, example_with_rels


def get_input_text(data):
    if data is not None:
        tokens = data['tokens']
        all_mentions = filter_non_events(data['allMentions'])
        all_mentions.sort(key=lambda x: x['tokens_ids'][0])
        all_mentions_text = [m['tokens'] for m in all_mentions]
        # all_ment_ids = [m['m_id'] for m in all_mentions]
        # all_pairs = data['allPairs']
        text = mark_events_in_text(tokens, all_mentions)
        print(f'The mentions are-{all_mentions_text}')
        print(f'The input text is-{text}')
        return text


def prepare_instructions(train_folder, dot_train_data, number_of_examples=1):
    examples = list()
    for file1 in os.listdir(train_folder):
        if len(examples) < number_of_examples:
            data = open_input_file(f'{train_folder}/{file1}')
            intput_text = get_input_text(data)
            output_example = dot_train_data[file1]['target']
            examples.append((intput_text, output_example))

    instruct_examples = list()
    for i, example in enumerate(examples):
        input = f'Example {i + 1}:\n' + example[0] + '\n'
        output = "Expected output:\n" + example[1] + "\n"
        instruct_examples.append(input + output)

    return "\n".join(instruct_examples)


def open_input_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def main(test_folder, train_folder, dot_test_data, dot_train_data, llm_to_use, instructions_func, output_file):
    # load json file
    predictions = dict()
    examples = prepare_instructions(train_folder, dot_train_data, 1)
    final_instructions = instructions_func(examples)
    count = 0
    for i, file1 in enumerate(os.listdir(test_folder)):
        if count >= 1:
            break
        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text = get_input_text(data)
        task_desc = final_instructions + '\n' + text
        response = llm_to_use(task_desc)
        predictions[file1] = {"target": response}

    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=4)


if __name__ == "__main__":
    _test_folder = 'data/MATRES/in_my_format/test'
    _dot_test_data = open_input_file('data/DOT_format/MATRES_test_dot.json')

    _train_folder = 'data/MATRES/in_my_format/train'
    _dot_train_data = open_input_file('data/DOT_format/MATRES_train_dot.json')

    _output_file = 'data/my_data/predictions/output/matres_gpt3.5_turbo_v2.json'

    _instructions = task_description_v2
    _llm_to_use = run_gpt3_5
    main(test_folder=_test_folder, train_folder=_train_folder, dot_test_data=_dot_test_data,
         dot_train_data=_dot_train_data, llm_to_use=_llm_to_use, instructions_func=_instructions, output_file=_output_file)
