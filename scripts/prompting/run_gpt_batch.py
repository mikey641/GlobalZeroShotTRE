import json
import os

import tiktoken
from openai import OpenAI

from scripts.prompting.prompts import task_description_v2
from scripts.prompting.run_llms import open_input_file, prepare_instructions, get_input_text


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


def from_dataset_to_batch_req(test_folder, train_folder, dot_train_data, instructions_func, output_file, num_of_files, num_of_examples, model_id):
    enc = tiktoken.encoding_for_model(model_id)
    json_lines = list()

    examples = prepare_instructions(train_folder, dot_train_data, num_of_examples)
    final_instructions = instructions_func(examples)

    count = 0
    for i, file1 in enumerate(os.listdir(test_folder)):
        if count == num_of_files:
            break

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text = get_input_text(data)
        prompt = final_instructions + '\n' + text

        # Count the number of tokens in the prompt
        print(f'Number of tokens in prompt={len(enc.encode(prompt))}')

        req = create_batch_request(prompt, file1, model_id)
        json_lines.append(req)

    with open(output_file, 'w', encoding='utf8') as file:
        for req in json_lines:
            file.write(json.dumps(req) + '\n')


def upload_batch_request(input_request_jsonl):
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(input_request_jsonl, 'rb'),
        purpose='batch'
    )

    print(f'Batch input file with id {batch_input_file.id} uploaded')


def run_batch(batch_input_file_id):
    client = OpenAI()
    create = client.batches.create(input_file_id=batch_input_file_id, endpoint="/v1/chat/completions",
                                   completion_window="24h", metadata={"description": "eventfull gpt4o batch job"})

    print(f'Batch object created: {create}')


def check_batch_status(batch_id, output_file):
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(f'Batch status: {batch}')

    if batch.status == 'completed':
        file_response = client.files.content(batch.output_file_id)
        with open(output_file, 'w', encoding='utf8') as file:
            for req in file_response.text.split('\n'):
                file.write(json.dumps(req) + '\n')
        print(file_response.text)


def convert_response(response_file, final_output_file):
    converted = dict()
    with (open(response_file, 'r', encoding='utf8') as file):
        for line in file:
            if not line.strip():
                continue

            response = json.loads(line)
            res_loads = json.loads(response)
            converted[res_loads['custom_id']] = {'target': res_loads['response']['body']['choices'][0]['message']['content']}

    with open(final_output_file, 'w', encoding='utf8') as file:
        json.dump(converted, file, indent=4)


if __name__ == "__main__":
    ############################################
    # This script needs to run sequentially, each time uncommenting the next step
    ############################################
    # --------------------------------------------
    # Step-1: create the request and save it to a file
    # --------
    # print("Running Step-1")
    # _test_folder = 'data/EventFullTrainExports/test'
    # _train_folder = 'data/EventFullTrainExports/dev'
    # _dot_train_data = open_input_file('data/DOT_format/EventFull_dev_dot.json')
    # _output_file = f'data/my_data/batch_req/eventfull_req_gpt3_5_3exmlps.jsonl'
    #
    # _instructions_func = task_description_v2
    # _num_of_files_to_prepare = 1
    # _num_of_examples = 3
    # _model_id = "gpt-3.5-turbo" #"gpt-4o"
    # from_dataset_to_batch_req(_test_folder, _train_folder, _dot_train_data, _instructions_func, _output_file, _num_of_files_to_prepare, _num_of_examples, _model_id)
    # --------------------------------------------

    # Step-2: upload the request to the server
    # --------
    # print("Running Step-2")
    # _input_request_jsonl = f'data/my_data/batch_req/eventfull_req_gpt3_5_2exmlps.jsonl'
    # upload_batch_request(_input_request_jsonl)
    # --------------------------------------------

    # Step-3: run the batch
    # --------
    # print("Running Step-3")
    # run_batch('file-qBdseRNxSc5sCwq2EeXVegHH')
    # --------------------------------------------

    # Step-4: check the status of the batch
    # --------
    print("Running Step-4")
    output_file = 'data/my_data/batch_req/eventfull_response_gpt3_5_2exmlps.jsonl'
    check_batch_status('batch_5D4oNvTjccCpm4Nj2J6NWMlc', output_file)
    # --------------------------------------------

    # Step-5: Convert response to DOT format response
    # --------
    print("Running Step-5")
    _output_file = 'data/my_data/batch_req/eventfull_response_gpt3_5_2exmlps.jsonl'
    _final_output_file = 'data/my_data/batch_req/eventfull_gpt3_5_-1pred_2exmples_task_description_v2.json'
    convert_response(_output_file, _final_output_file)
    # --------------------------------------------
    print("Done!")
