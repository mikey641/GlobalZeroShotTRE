import json
import os

import tiktoken

from scripts.prompting_global.prompt_old_bkp import task_description_v2
from scripts.prompting_global.run_gpt_batch import create_batch_request
from scripts.prompting_global.run_llms import gpt4o, prepare_instructions, get_input_text
from scripts.utils.io_utils import open_input_file


def from_dataset_to_batch_req(test_folder, train_folder, dot_train_data, instructions_func, output_file, num_of_files,
                              num_of_examples, sel_random, selected_file, model_id):
    enc = tiktoken.encoding_for_model(model_id)
    json_lines = list()

    examples = prepare_instructions(train_folder, dot_train_data, num_of_examples, sel_random, selected_file)
    final_instructions = instructions_func(examples)

    total_tokens = 0
    count = 0
    for i, file1 in enumerate(os.listdir(test_folder)):
        if count == num_of_files:
            break

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text, all_pairs = get_input_text(data)
        prompt = final_instructions + '\n' + text

        # Count the number of tokens in the prompt
        encoded_text = enc.encode(prompt)
        total_tokens += len(encoded_text)
        print(f'Number of tokens in prompt={len(encoded_text)}')

        req = create_batch_request(prompt, file1, model_id)
        json_lines.append(req)

    print(f'Total number of tokens in all prompts={total_tokens}')
    split = 1
    if total_tokens > 90000:
        if total_tokens / 2 < 90000:
            split = 2
        elif total_tokens / 3 < 90000:
            split = 3
        else:
            split = 4

    chunk_size = len(json_lines) // split
    chunks = [json_lines[i:i + chunk_size] for i in range(0, len(json_lines), chunk_size)]
    for i in range(len(chunks)):
        with open(f'{output_file}_chunk{i}.jsonl', 'w', encoding='utf8') as file1:
            for req in chunks[i]:
                file1.write(json.dumps(req) + '\n')


if __name__ == "__main__":
    _instructions_func = task_description_v2
    _model_id = 'gpt-4o'

    _train_folder = 'data/OmniTemp/dev'
    _all_data = open_input_file('data/DOT_format/EventFull_dev_dot.json')

    enc = tiktoken.encoding_for_model(_model_id)
    total_tokens = 0
    json_lines = list()
    for doc_name_train in _all_data.keys():
        _prep_instructions = prepare_instructions(_train_folder, _all_data, number_of_examples=1, sel_random=False, selected_file=doc_name_train)
        _final_instructions = _instructions_func(_prep_instructions)
        _output_file = f'data/my_data/k_fold/{doc_name_train}_req_cross_valid.jsonl'
        for doc_name_test in _all_data.keys():
            if doc_name_train == doc_name_test:
                continue

            data = open_input_file(f'{_train_folder}/{doc_name_test}')
            text, all_pairs = get_input_text(data)
            prompt = _final_instructions + '\n' + text

            # Count the number of tokens in the prompt
            encoded_text = enc.encode(prompt)
            total_tokens += len(encoded_text)
            print(f'Number of tokens in prompt={len(encoded_text)}')

            req = create_batch_request(prompt, doc_name_test, _model_id)
            json_lines.append(req)

        with open(_output_file, 'w', encoding='utf8') as file:
            for req in json_lines:
                file.write(json.dumps(req) + '\n')
            json_lines.clear()
