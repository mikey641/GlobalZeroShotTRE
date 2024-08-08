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


def main():
    # load json file
    results_to_print = list()
    for file1 in os.listdir(input_folder):
        with open(f'{input_folder}/{file1}') as file:
            data = json.load(file)

        tokens = data['tokens']
        all_mentions = filter_non_events(data['allMentions'])
        all_mentions.sort(key=lambda x: x['tokens_ids'][0])
        all_mentions_text = [m['tokens'] for m in all_mentions]
        all_ment_ids = [m['m_id'] for m in all_mentions]
        all_pairs = data['allPairs']
        text = mark_events_in_text(tokens, all_mentions)

        # num_of_mentions_txt = f'The following text contains-{len(all_mentions)} mentions \n'
        # print(num_of_mentions_txt)
        print(f'The mentions are-{all_mentions_text}')
        print(f'The input text is-{text}')
        all_prompt_pairs, example_with_rels = get_prompt_pairs(all_mentions, all_pairs)
        # print(f'The pairs are-{all_prompt_pairs}')

        # get_example_matrix(all_pairs, all_ment_ids)
        task_desc = instructions() + text + '\n' + indicate_pairs + "\n" + ",".join(all_prompt_pairs)
        response = llm_to_use(task_desc)
        results_to_print.append(f'File:{file1}\n')
        results_to_print.append(f'Gold:{",".join(example_with_rels)}\n')
        results_to_print.append(f'Pred:{response}\n')
        results_to_print.append(f'---------------\n')

        print(f'{file1} Gold pairs: {example_with_rels}')
        print(f'{file1} Predicted pairs: {response}')
        print("-----------------------------------")

    with open(output_file, 'w') as file:
        file.writelines(results_to_print)


if __name__ == "__main__":
    indicate_pairs = "Pairs to Resolve:"
    input_folder = 'data/my_data/predictions/input'
    output_file = 'data/my_data/predictions/output/gpt4_turbo_v2.txt'

    instructions = task_description_v2
    llm_to_use = run_gpt4_turbo
    main()
