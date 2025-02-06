import json
import os
import random
import tiktoken

from scripts.prompting.prompts import task_description_4res_only_global, task_description_6res_only_global, \
    task_description_4res_only_timeline, task_description_6res_only_timeline, task_description_6rels


def create_batch_request(prompt, request_id, model_id):
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4096
        }
    }


def mark_events_in_text(tokens, all_mentions):
    for mention in all_mentions:
        tok_first_id = mention['tokens_ids'][0]
        tok_last_id = mention['tokens_ids'][-1]
        tokens[tok_first_id] = f'<{tokens[tok_first_id]}'
        tokens[tok_last_id] = f'{tokens[tok_last_id]}({mention["m_id"]})>'
    return " ".join(tokens)


def filter_non_events(events):
    return [e for e in events if e['axisType'] == 'main']


def get_input_text(data):
    if data is not None:
        tokens = data['tokens']
        all_mentions = filter_non_events(data['allMentions'])
        all_mentions.sort(key=lambda x: x['tokens_ids'][0])
        # all_mentions_text = [m['tokens'] for m in all_mentions]
        # all_ment_ids = [m['m_id'] for m in all_mentions]
        all_pairs = data['allPairs']
        text = mark_events_in_text(tokens, all_mentions)
        # print(f'The mentions are-{all_mentions_text}')
        # print(f'The input text is-{text}')
        return text, all_pairs


def open_input_file(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def convert_response(response_file, final_output_file):
    converted = dict()
    with (open(response_file, 'r', encoding='utf8') as file1):
        for line in file1:
            if not line.strip():
                continue

            response = json.loads(line)
            res_loads = json.loads(response)
            converted[res_loads['custom_id']] = {'target': res_loads['response']['body']['choices'][0]['message']['content']}

    with open(final_output_file, 'w', encoding='utf8') as file2:
        json.dump(converted, file2, indent=4)


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


def get_example(file_to_use, target, reduction):
    data = open_input_file(file_to_use)
    intput_text, all_pairs = get_input_text(data)
    if reduction > 0:
        split_out = target.split('\n')
        target_pref = split_out[0:2]
        target_suffix = split_out[-2:]
        new_target = split_out[2:-3]
        sample_size = max(1, int(len(new_target) * reduction))
        indices_to_remove = random.sample(range(len(new_target)), sample_size)
        output_example = [new_target[i] for i in range(len(new_target)) if i not in indices_to_remove]
        # output_example = random.sample(new_target, sample_size)
        output_example = target_pref + output_example + target_suffix
        output_example = '\n'.join(output_example)
    else:
        output_example = target
    return intput_text, output_example


def get_reverse_label(label):
    if label == 'before':
        return 'after'
    elif label == 'after':
        return 'before'
    elif label == 'is_included':
        return 'includes'
    elif label == 'includes':
        return 'is_included'
    else:
        return label

def arrange_pairs(all_pairs, ment_dict):
    for pair in all_pairs:
        m1 = ment_dict[pair['_firstId']]
        m2 = ment_dict[pair['_secondId']]
        if m1['tokens_ids'][0] > m2['tokens_ids'][0]:
            pair['_firstId'], pair['_secondId'] = pair['_secondId'], pair['_firstId']
            pair['_relation'] = get_reverse_label(pair['_relation'])


def get_all_pairs(all_pairs, ment_dict, reduction):
    ret_pairs_dot = list()
    if reduction > 0:
        sample_size = max(1, int(len(all_pairs) * reduction))
        indices_to_remove = random.sample(range(len(all_pairs)), sample_size)
        all_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in indices_to_remove]

    # arrange_pairs(all_pairs, ment_dict)
    pairs_with_id = list()
    for pair in all_pairs:
        new_pair = pair.copy()
        new_pair['index'] = ment_dict[pair['_firstId']]['tokens_ids'][0]
        pairs_with_id.append(new_pair)

    sorted_pairs = sorted(pairs_with_id, key=lambda x: x['index'])

    for pair in sorted_pairs:
        # m1 = f"{m['tokens']}({m['m_id']})"
        m1 = ment_dict[pair['_firstId']]
        m2 = ment_dict[pair['_secondId']]
        first_ment = f"{m1['tokens']}({m1['m_id']})"
        second_ment = f"{m2['tokens']}({m2['m_id']})"
        ret_pairs_dot.append(f"{first_ment} -- {second_ment}")

    return ret_pairs_dot


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


def upload_batch_request(client, input_request_jsonl):
    batch_input_file = client.files.create(
        file=open(input_request_jsonl, 'rb'),
        purpose='batch'
    )

    print(f'Batch input file with id {batch_input_file.id} uploaded')
    return batch_input_file.id


def run_batch(client, batch_input_file_id):
    create = client.batches.create(input_file_id=batch_input_file_id, endpoint="/v1/chat/completions",
                                   completion_window="24h", metadata={"description": "eventfull gpt4o batch job"})

    print(f'Batch object created: {create}')
    return create.id


def check_batch_status(client, batch_id, output_file):
    batch = client.batches.retrieve(batch_id)
    print(f'Batch status: {batch}')

    if batch.status == 'completed':
        file_response = client.files.content(batch.output_file_id)
        with open(output_file, 'w', encoding='utf8') as file:
            for req in file_response.text.split('\n'):
                file.write(json.dumps(req) + '\n')
        print(file_response.text)