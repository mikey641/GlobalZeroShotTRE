import json
import string
import time

from tqdm import tqdm

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


def run_CoT(all_examples, llm_to_use):
    predictions = {}
    for i, example in enumerate(tqdm(all_examples)):
        # if i == 3:
        #     break

        on_file = example['file']
        source = example['source']
        source_text = example['source_text']
        target = example['target']
        target_text = example['target_text']
        instruction = example['instruct']
        gold_label = example['gold_label']
        key = f"{on_file}#{source}#{target}"

        llm_to_use.clear()

        try:
            response = llm_to_use.run_model_chat(instruction)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                is_same = True
            else:
                is_same = False

            response = llm_to_use.run_model_chat(get_equal_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'equal', "gold_label": gold_label}
                continue

            response = llm_to_use.run_model_chat(get_before_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'before', "gold_label": gold_label}
                continue

            response = llm_to_use.run_model_chat(get_after_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'after', "gold_label": gold_label}
                continue

            response = llm_to_use.run_model_chat(get_is_included_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'is_included', "gold_label": gold_label}
                continue

            response = llm_to_use.run_model_chat(get_includes_prompt(source, source_text, target, target_text, same_prompt=is_same))
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'includes', "gold_label": gold_label}
                continue

            predictions[key] = {"target": 'vague', "gold_label": gold_label}
        except Exception as e:
            predictions[key] = {"target": "Generation Failed", "gold_label": gold_label}
            print('Failed to predict', repr(e))

    return predictions


def run_zero_shot(all_examples, llm_to_use):
    predictions = {}
    for i, example in enumerate(tqdm(all_examples)):
        # if i == 3:
        #     break

        on_file = example['file']
        source = example['source']
        target = example['target']
        instruction = example['instruct']
        gold_label = example['gold_label']
        key = f"{on_file}#{source}#{target}"

        try:
            response = llm_to_use(instruction)
            predictions[key] = {"target": response, "gold_label": gold_label}
        except Exception as e:
            print('Failed to predict', repr(e))
            predictions[key] = {"target": "Generation Failed", "gold_label": gold_label}
            return None

    return predictions


if __name__ == "__main__":
    # read all line from file
    # _llm_to_use = TogetherModel('meta-llama/Llama-3.3-70B-Instruct-Turbo')
    _llm_to_use = GeminiChatModel('models/gemini-2.0-flash')
    _test_set = 'nt'

    with open("data/my_data/zero_shot/nt_6rel_cot_prompts.jsonl") as _file:
        data = json.load(_file)

    start_time = time.time()
    _predictions = run_CoT(data, _llm_to_use)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    with open(f"data/my_data/zero_shot/new_expr/{_test_set}_{_llm_to_use.get_model_name()}_cot_predictions.json", "w") as _file:
        json.dump(_predictions, _file, indent=4)
