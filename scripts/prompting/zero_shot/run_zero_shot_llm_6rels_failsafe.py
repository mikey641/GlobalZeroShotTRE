import json
import re
import string
import time

from tqdm import tqdm

from scripts.prompting.zero_shot.run_zero_shot_llm_failsafe import write_json_line, from_jsonl_to_key_dict
from scripts.utils.io_utils import load_json_lines
from scripts.utils.llms_definitions import TogetherModel, GeminiChatModel


def get_equal_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""Did <EVENT {source_id}>{source_text}</EVENT> and <EVENT {target_id}>{target_text}</EVENT> simultaneously happened{same_event}? Answer with yes or no only."""


def get_before_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> before <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer with yes or no only."""


def get_after_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> after <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer with yes or no only."""


def get_is_included_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> included in <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer with yes or no only."""


def get_includes_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> includes <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer with yes or no only."""


def run_CoT(all_examples, llm_to_use, key_set, output_file_stream):
    for i, example in enumerate(tqdm(all_examples)):
        # if i == 10:
        #     break

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
                write_json_line(prediction, output_file_stream)
                continue

            response = llm_to_use.run_model_chat(get_is_included_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                prediction = {"key": key, "target": 'is_included', "gold_label": gold_label}
                write_json_line(prediction, output_file_stream)
                continue

            response = llm_to_use.run_model_chat(get_includes_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                prediction = {"key": key, "target": 'includes', "gold_label": gold_label}
            else:
                prediction = {"key": key, "target": 'vague', "gold_label": gold_label}
        except Exception as e:
            prediction = {"key": key, "target": "Generation Failed", "gold_label": gold_label}
            print('Failed to predict', repr(e))

        write_json_line(prediction, output_file_stream)


if __name__ == "__main__":
    # read all line from file
    # _llm_to_use = TogetherModel('meta-llama/Llama-3.3-70B-Instruct-Turbo')
    _llm_to_use = TogetherModel('deepseek-ai/DeepSeek-R1')
    # _llm_to_use = GeminiChatModel('models/gemini-2.0-flash')
    _input_file = "data/my_data/zero_shot/nt_6rel_cot_prompts.jsonl"

    _test_set = 'nt'
    _output_file = f"data/my_data/zero_shot/new_expr/{_test_set}_{_llm_to_use.get_model_name()}_{run_CoT.__name__}_predictions.jsonl"

    print(f"Using LLM: {_llm_to_use.get_model_name()}")
    print(f"Using input file: {_input_file}")
    print(f"Using test set: {_test_set}")
    print("Running CoT for 6 relations dataset!")

    with open(_input_file) as _file:
        data = json.load(_file)

    _loaded_data = load_json_lines(_output_file)
    _key_set = from_jsonl_to_key_dict(_loaded_data)

    start_time = time.time()
    with open(_output_file, "a") as _ofs:
        run_CoT(data, _llm_to_use, _key_set, _ofs)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
