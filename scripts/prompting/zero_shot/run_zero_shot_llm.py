import json
import re
import string
import time

from tqdm import tqdm

from scripts.prompting.run_llms import gpt4o, gpt4, gpt3_5


def get_equal_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""Did <EVENT {source_id}>{source_text}</EVENT> and <EVENT {target_id}>{target_text}</EVENT> simultaneously happened{same_event}? Answer yer or no."""


def get_before_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> before <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer yer or no."""


def get_after_prompt(source_id, source_text, target_id, target_text, same_prompt=True):
    if same_prompt:
        same_event = " in that event"
    else:
        same_event = ""
    return f"""is <EVENT {source_id}>{source_text}</EVENT> after <EVENT {target_id}>{target_text}</EVENT>{same_event}? Answer yer or no."""


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

        messages = [
            {
                "role": "user",
                "content": instruction
            },
        ]

        try:
            response = llm_to_use(None, messages)
            messages.append({"role": "assistant", "content": response})
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                is_same = True
            else:
                is_same = False

            messages.append({"role": "user", "content": get_equal_prompt(source, source_text, target, target_text, same_prompt=is_same)})

            response = llm_to_use(None, messages)
            messages.append({"role": "assistant", "content": response})
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'equal', "gold_label": gold_label}
                continue
            else:
                messages.append({"role": "user", "content": get_before_prompt(source, source_text, target, target_text, same_prompt=is_same)})
            response = llm_to_use(None, messages)
            messages.append({"role": "assistant", "content": response})
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'before', "gold_label": gold_label}
                continue
            else:
                messages.append({"role": "user", "content": get_after_prompt(source, source_text, target, target_text, same_prompt=is_same)})
            response = llm_to_use(None, messages)
            response = response.rstrip(string.whitespace + string.punctuation).lower()

            if 'yes' in response:
                predictions[key] = {"target": 'after', "gold_label": gold_label}
            else:
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


def run_first_timeline(all_examples, llm_to_use):
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

        pattern = r"[Rr]elation\s?=\s?([A-Za-z]+)"
        try:
            response = llm_to_use(instruction)
            match = re.search(pattern, response)
            if match:
                value = match.group(1)
                predictions[key] = {"target": value, "gold_label": gold_label}
            else:
                predictions[key] = {"target": response, "gold_label": gold_label}
        except Exception as e:
            print('Failed to predict', repr(e))
            predictions[key] = {"target": "Generation Failed", "gold_label": gold_label}
            return None

    return predictions



if __name__ == "__main__":
    # read all line from file
    _llm_to_use = gpt4o

    with open("data/my_data/zero_shot/matres_first_timeline_brief_prompts.jsonl") as _file:
        data = json.load(_file)

    start_time = time.time()
    _predictions = run_first_timeline(data, _llm_to_use)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    with open(f"data/my_data/zero_shot/matres_{_llm_to_use.__name__}_first_timeline_brief_predictions.json", "w") as _file:
        json.dump(_predictions, _file, indent=4)
