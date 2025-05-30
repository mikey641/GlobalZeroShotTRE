import json
import re
import string
import time

from tqdm import tqdm

from scripts.utils.io_utils import load_json_lines, write_json_line
from scripts.utils.llms_definitions import GPTModel, GeminiChatModel, TogetherModel


def get_equal_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""Did <EVENT {source_id}>{source_text}</EVENT> and <EVENT {target_id}>{target_text}</EVENT> simultaneously happened{same_event}? Answer yes or no."""


def get_before_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> before <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer yes or no."""


def get_after_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> after <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer yes or no."""


def run_CoT(all_examples, llm_to_use, key_set, output_file_stream, target_file_stream):
    for i, example in enumerate(tqdm(all_examples)):
        if i == 108:
            break

        on_file = example['file']
        source = example['source']
        source_text = example['source_text']
        target = example['target']
        target_text = example['target_text']
        instruction = example['instruct']
        gold_label = example['gold_label']
        key = f"{on_file}#{source}#{target}"

        if key in key_set:
            continue

        if len(llm_to_use.messages) > 0:
            write_json_line(llm_to_use.messages, target_file_stream)

        llm_to_use.clear()

        try:
            response = llm_to_use.run_model_chat(instruction)
            # to clean DeepSeek response
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                is_same = True
            else:
                is_same = False

            response = llm_to_use.run_model_chat(get_equal_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                prediction = {"key": key, "target": 'equal', "gold_label": gold_label}
                write_json_line(prediction, output_file_stream)
                continue

            response = llm_to_use.run_model_chat(get_before_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                prediction = {"key": key, "target": 'before', "gold_label": gold_label}
                write_json_line(prediction, output_file_stream)
                continue

            response = llm_to_use.run_model_chat(get_after_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                prediction = {"key": key, "target": 'after', "gold_label": gold_label}
            else:
                prediction = {"key": key, "target": 'vague', "gold_label": gold_label}
        except Exception as e:
            prediction = {"key": key, "target": "Generation Failed", "gold_label": gold_label}
            print('Failed to predict', repr(e))

        write_json_line(prediction, output_file_stream)


def from_jsonl_to_key_dict(loaded_data):
    """
    Convert loaded data from jsonl to dict
    """
    keys = set()
    for line in loaded_data:
        keys.add(line['key'])

    return keys


if __name__ == "__main__":
    '''
    This is the script to run the 4 relation CoT
    To run the 6 relation CoT, run the script *run_zero_shot_llm_6rel.py*
    '''

    # read all line from file
    # _llm_to_use = GPTModel('gpt-4o')
    # _llm_to_use = GeminiChatModel('gemini-2.0-flash')
    # _llm_to_use = TogetherModel('meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')
    # _llm_to_use = TogetherModel('meta-llama/Llama-3.3-70B-Instruct-Turbo-Free')
    # _llm_to_use = TogetherModel('google/gemma-2b-it')
    _llm_to_use = TogetherModel('deepseek-ai/DeepSeek-R1')
    _test_set = 'omni'

    _input_file = "data/my_data/zero_shot/eventfull_cot_prompts.jsonl"
    _output_file = f"data/my_data/zero_shot/log/{_test_set}_{_llm_to_use.get_model_name()}_{run_CoT.__name__}_predictions.jsonl"
    _output_targets = f"data/my_data/zero_shot/log/{_test_set}_{_llm_to_use.get_model_name()}_{run_CoT.__name__}_targets.jsonl"

    print(f"Using LLM: {_llm_to_use.get_model_name()}")
    print(f"Using input file: {_input_file}")
    print(f"Using output file: {_output_file}")
    print(f"Using test set: {_test_set}")
    print("Running CoT for 4 relations dataset!")
    print(f'running method: {run_CoT.__name__}()')

    with open(_input_file) as _file:
        data = json.load(_file)

    _loaded_data = load_json_lines(_output_file)
    _key_set = from_jsonl_to_key_dict(_loaded_data)

    start_time = time.time()
    with open(_output_file, "a") as _ofs:
        with open(_output_targets, "a") as _tar_ofs:
            run_CoT(data, _llm_to_use, _key_set, _ofs, _tar_ofs)
    end_time = time.time()

    execution_time = end_time - start_time
    print()
    print(f"Execution time: {execution_time:.4f} seconds")
