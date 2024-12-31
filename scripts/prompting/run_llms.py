import json
import os
import random

import google.generativeai as genai
from openai import OpenAI
from together import Together
from tqdm import tqdm

from scripts.eval.dataset.utils import find_ment_by_id
from scripts.prompting.jup_utils import open_input_file, get_input_text, prepare_instructions
from scripts.prompting.prompts import task_description_v2, task_description_v3

gemini_pro_model = None
gemini_flash_model = None


def llama_8b(_prompt):
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    print(f"Prompt: {_prompt}")

    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": _prompt}],
        max_tokens=4096,
        stream=True,
    )

    response = []
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
        if chunk.choices[0].delta.content:
            response.append(chunk.choices[0].delta.content)

    return "".join(response)


def gpt4_turbo(_prompt):
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


def gpt4o_mini(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def gpt4o(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def gpt4(_prompt):
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


def gpt3_5(_prompt):
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
    global gemini_pro_model
    if gemini_pro_model is None:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_pro_model = genai.GenerativeModel('gemini-1.5-pro')
        model_info = genai.get_model("models/gemini-1.5-pro")
        print(f"{model_info.input_token_limit=}")
        print(f"{model_info.output_token_limit=}")

    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)
    print('prompt text:\n', _prompt)
    print("prompt: ", gemini_pro_model.count_tokens(_prompt))

    response = gemini_pro_model.generate_content(_prompt)
    print("response: ", gemini_pro_model.count_tokens(response.text))
    return response.text


def run_gemini_flash(_prompt):
    global gemini_flash_model
    if gemini_flash_model is None:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash')
        model_info = genai.get_model("models/gemini-1.5-flash")
        print(f"{model_info.input_token_limit=}")
        print(f"{model_info.output_token_limit=}")

    print('prompt text:\n', _prompt)
    print("prompt: ", gemini_flash_model.count_tokens(_prompt))
    response = gemini_flash_model.generate_content(_prompt)
    print("response: ", gemini_flash_model.count_tokens(response.text))
    return response.text


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


def main(test_folder, train_folder, dot_train_data, llm_to_use, instructions_func, output_file,
         number_of_pred, prompt_examples, selected_file, reduction=-1.0):
    # load json file
    predictions = dict()
    examples = prepare_instructions(train_folder, dot_train_data, prompt_examples, selected_file, reduction)
    final_instructions = instructions_func(examples)
    count = 0
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        if count == number_of_pred:
            break

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text = get_input_text(data)
        task_desc = final_instructions + '\n' + text
        try:
            response = llm_to_use(task_desc)
            predictions[file1] = {"target": response}
        except Exception as e:
            predictions[file1] = {"target": "Generation Failed"}

    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=4)


if __name__ == "__main__":
    example_db = 'eventfull'
    test_db = 'matres'
    # -1 for all predictions
    # Number of prompt examples
    num_of_pred = -1
    num_of_prompt_examples = 1
    _reduction = -1
    # APW19980213.1380.json
    _selected_file = '21_final.json'
    _instructions = task_description_v2
    _llm_to_use = gpt4o
    _test_folder = 'data/MATRES/in_my_format/test'
    # _dot_test_data = open_input_file('data/DOT_format/MATRES_test_dot.json')
    # _test_folder = 'data/EventFullTrainExports/test'

    # _train_folder = 'data/MATRES/in_my_format/train'
    # _dot_train_data = open_input_file('data/DOT_format/MATRES_train_dot.json')
    _train_folder = 'data/EventFullTrainExports/train'
    _dot_train_data = open_input_file('data/DOT_format/EventFull_dev_dot.json')

    # _output_file = f'data/my_data/predictions/{test_db}/{example_db}_{_llm_to_use.__name__}_{num_of_pred}pred_{num_of_prompt_examples}exmples_{_instructions.__name__}.json'
    _output_file = f'data/my_data/predictions/{test_db}/outputs/100prs_{example_db}_{_llm_to_use.__name__}_{num_of_pred}pred_{num_of_prompt_examples}exmp_rand_30_{_instructions.__name__}.json'

    main(test_folder=_test_folder, train_folder=_train_folder,
         dot_train_data=_dot_train_data, llm_to_use=_llm_to_use, instructions_func=_instructions,
         output_file=_output_file, number_of_pred=num_of_pred, prompt_examples=num_of_prompt_examples,
         selected_file=_selected_file, reduction=_reduction)
