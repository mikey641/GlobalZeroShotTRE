import json


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