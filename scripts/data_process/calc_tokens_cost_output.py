import os

import tiktoken
from tqdm import tqdm

from scripts.prompting.global_timeline_consistency.prompts import task_description_6res_only_global, \
    task_description_4res_only_global, task_description_4res_only_timeline, task_description_6res_only_timeline
from scripts.prompting.global_timeline_consistency.run_llms import get_complete_prompt
from scripts.utils.io_utils import open_input_file


def main(output_folder):
    # llms_names = ['gpt-4o', 'Llama-70', 'Llama-405', 'DeepSeek']
    # cost_per_1m_out_token = [10/1000000.0, 0.88/1000000.0, 3.5/1000000.0, 7/1000000.0]

    llms_names = ['DeepSeek']
    cost_per_1m_out_token = [7/1000000.0]

    enc = tiktoken.encoding_for_model("gpt-4o")
    input_tokens = 0
    for i, file1 in enumerate(sorted(os.listdir(output_folder))):
        if file1.endswith('.json'):
            parsed_output = open_input_file(f'{output_folder}/{file1}')
            for doc_id in parsed_output:
                target = parsed_output[doc_id]['target']
                input_tokens += len(enc.encode(target))
        elif file1.endswith('.jsonl'):
            with open(f'{output_folder}/{file1}', 'r', encoding='utf-8') as file:
                # read file line by line
                for line in file:
                    parsed_output = line.strip()
                    input_tokens += len(enc.encode(parsed_output))

        for x in range(len(llms_names)):
            print(f'output: {file1}: '
                  f'tokens:{input_tokens}: '
                  f'output_cost-{llms_names[x]}:{input_tokens * cost_per_1m_out_token[x]}')

        input_tokens = 0


if __name__ == "__main__":
    _output_folder = 'data/my_data/zero_shot/log'
    main(_output_folder)

