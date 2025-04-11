import os

import tiktoken
from tqdm import tqdm

from scripts.prompting.global_timeline_consistency.prompts import task_description_6res_only_global, \
    task_description_4res_only_global, task_description_4res_only_timeline, task_description_6res_only_timeline
from scripts.prompting.global_timeline_consistency.run_llms import get_complete_prompt
from scripts.utils.io_utils import open_input_file


def main(datasets):
    instruct_funcs = [task_description_4res_only_global,
                      task_description_6res_only_global,
                      task_description_4res_only_timeline,
                      task_description_6res_only_timeline]

    llms_names = ['gpt-4o', 'Llama-70', 'Llama-405', 'DeepSeek']
    cost_per_1m_in_token = [2.5/1000000.0, 0.88/1000000.0, 3.5/1000000.0, 3/1000000.0]
    cost_per_1m_out_token = [10/1000000.0, 0.88/1000000.0, 3.5/1000000.0, 7/1000000.0]

    enc = tiktoken.encoding_for_model("gpt-4o")
    input_tokens = 0
    for data in datasets:
        for prompt_desc in instruct_funcs:
            for i, file1 in enumerate(os.listdir(data)):
                parsed_input = open_input_file(f'{data}/{file1}')
                task_desc = get_complete_prompt(parsed_input, prompt_desc, gen_pairs=True)
                input_tokens += len(enc.encode(task_desc))

            for x in range(len(llms_names)):
                print(f'Input: Dataset-{data}: '
                      f'prompt-{prompt_desc.__name__}: tokens:{input_tokens}: '
                      f'input_cost-{llms_names[x]}:{input_tokens * cost_per_1m_in_token[x]}')
            # print('-------------')
            input_tokens = 0


if __name__ == "__main__":
    _omni_input = 'data/OmniTemp/test'
    _tbd_input = 'data/TimeBank-Dense/test_converted_allpairs_chunked'
    _matres_input = 'data/MATRES/in_my_format/test_all_pairs_chunked'
    _nt_input = 'data/NarrativeTime_A1/converted_no_overlap/test_18ment'

    _omni_deepseek_out_timeline = 'data/my_data/predictions/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_0.json'
    main([_omni_input, _tbd_input, _matres_input, _nt_input])

