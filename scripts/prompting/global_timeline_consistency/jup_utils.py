import json
import os
import random
import tiktoken

from scripts.prompting.global_timeline_consistency.prompts_gpt import task_description_6res_only_global
from scripts.utils.gpt_utils import create_batch_request
from scripts.utils.io_utils import open_input_file
from scripts.utils.omni_format_utils import get_input_text, get_example, get_all_pairs


def prepare_instructions(train_folder, dot_train_data, number_of_examples=1, selected_file=None, reduction=-1):
    examples = list()
    if not selected_file:
        file1 = random.choice(os.listdir(train_folder))
        print(f'Randomly Selected File: {file1}')
        intput_text, output_example = get_example(f'{train_folder}/{file1}', dot_train_data[file1]['target'], reduction)
        examples.append((intput_text, output_example))
    else:
        for file1 in os.listdir(train_folder):
            if file1 != selected_file:
                continue

            print(f'Using non random file: {file1}')
            if len(examples) < number_of_examples:
                print(f'Sequentially Selected File: {file1}')
                intput_text, output_example = get_example(f'{train_folder}/{file1}', dot_train_data[file1]['target'], reduction)
                examples.append((intput_text, output_example))

    instruct_examples = list()
    for i, example in enumerate(examples):
        input_expl = f'Example {i + 1}:\n' + example[0] + '\n'
        output_expl = "Expected output:\n" + example[1] + "\n"
        instruct_examples.append(input_expl + output_expl)

    return "\n".join(instruct_examples)


def from_dataset_to_batch_req(test_folder, train_folder, dot_train_data, output_file, num_of_files,
                              num_of_examples, selected_file, model_id, reduction=-1, gen_pairs=False):
    enc = tiktoken.encoding_for_model(model_id)
    json_lines = list()

    examples = prepare_instructions(train_folder, dot_train_data, num_of_examples, selected_file, reduction)
    final_instructions = task_description_6res_only_global(examples)

    total_tokens = 0
    count = 0
    for i, file1 in enumerate(os.listdir(test_folder)):
        if count == num_of_files:
            break

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text, all_pairs = get_input_text(data)
        prompt = final_instructions + '\n' + text

        if gen_pairs:
            all_mentions = data['allMentions']
            all_ment_ids = {m['m_id']: m for m in all_mentions}
            example_matrix = get_all_pairs(all_pairs, all_ment_ids, reduction)
            prompt += '\nPairs require classification:\n' + '\n'.join(example_matrix)

        # Count the number of tokens in the prompt
        encoded_text = enc.encode(prompt)
        total_tokens += len(encoded_text)
        print(f'Number of tokens in prompt={len(encoded_text)}')

        req = create_batch_request(prompt, file1, model_id)
        json_lines.append(req)

    print(f'Total number of tokens in all prompts={total_tokens}')
    split = 1
    curr_tok_limit = 800000
    if total_tokens > curr_tok_limit:
        if total_tokens / 2 < curr_tok_limit:
            split = 2
        elif total_tokens / 3 < curr_tok_limit:
            split = 3
        else:
            split = 4

    chunk_size = len(json_lines) // split
    chunks = [json_lines[i:i + chunk_size] for i in range(0, len(json_lines), chunk_size)]
    for i in range(len(chunks)):
        with open(f'{output_file}_chunk{i}.jsonl', 'w', encoding='utf8') as file1:
            for req in chunks[i]:
                file1.write(json.dumps(req) + '\n')
