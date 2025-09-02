import json
import os
import sys
import time

import argparse
from tqdm import tqdm

from scripts.run.prompts import task_description_6res_only_global, task_description_4res_only_global, \
    task_description_6res_only_timeline, task_description_4res_only_timeline
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel, GPTModel, GeminiModel
from scripts.utils.omni_format_utils import get_input_text, get_all_pairs


def get_complete_prompt(data, instructions_func, gen_pairs):
    text, all_pairs = get_input_text(data)
    final_instructions = instructions_func()
    task_desc = final_instructions + '\n' + text
    if gen_pairs:
        all_mentions = data['allMentions']
        all_ment_ids = {m['m_id']: m for m in all_mentions}
        example_matrix = get_all_pairs(all_pairs, all_ment_ids, reduction=-1)
        task_desc += '\nPairs require classification:\n' + '\n'.join(example_matrix)

    return task_desc


def main(test_folder, llm_to_use, instructions_func, output_file,
         selected_file, gen_pairs=True):
    # load json file
    predictions = dict()
    count = 0
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        if selected_file is not None and file1 != selected_file:
            continue

        if not file1.endswith('.json') and not file1.endswith('.jsonl'):
            continue

        print("Processing file:", file1)

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        task_desc = get_complete_prompt(data, instructions_func, gen_pairs)

        try:
            response = llm_to_use.run_model(task_desc)
            predictions[file1] = {"target": response}
        except Exception as e:
            predictions[file1] = {"target": "Generation Failed"}

    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run llm on test data")
    parser.add_argument("--test_ds", help="The test dataset name (nt, matres, omni, tbd)", type=str, required=True)
    parser.add_argument("--instruct", help="Should be global/timeline", type=str, required=True)
    parser.add_argument("--api", help="Should be together/gpt/gemini", type=str, required=True)
    parser.add_argument("--model_name", help="the model to use", type=str, required=True)
    parser.add_argument("--repeat", help="Times to repeat the experiment", type=int, required=True)
    parser.add_argument("--selected_file", help="To only run a single specific file", type=str, required=False)
    parser.add_argument("--output_folder", help="Where to save the outputs", type=str, required=False)

    args = parser.parse_args()
    print(vars(args))

    num_of_repetitions = args.repeat
    _selected_file = args.selected_file

    if args.test_ds == "nt":
        _test_folder = 'data/NT_6/test_18ment'
    elif args.test_ds == "matres":
        _test_folder = 'data/MATRES/test_all_pairs_chunked'
    elif args.test_ds == "omni":
        _test_folder = 'data/OmniTemp/test'
    elif args.test_ds == "tbd":
        _test_folder = 'data/TBD/test_converted_allpairs_chunked'
    elif args.test_ds == "maven":
        _test_folder = 'data/MAVEN-ERE/valid'
    else:
        raise ValueError("Invalid test database name.")

    if args.instruct == 'global' and args.test_ds in ["nt", "tbd"]:
        _instructions = task_description_6res_only_global
    elif args.instruct == 'global' and args.test_ds in ["matres", "omni"]:
        _instructions = task_description_4res_only_global
    elif args.instruct == 'timeline' and args.test_ds in ["nt", "tbd"]:
        _instructions = task_description_6res_only_timeline
    elif args.instruct == 'timeline' and args.test_ds in ["matres", "omni", "maven",]:
        _instructions = task_description_4res_only_timeline
    else:
        raise ValueError("Invalid instruction type.")

    if args.api == 'together':
        _llm_to_use = TogetherModel(args.model_name)
    elif args.api == 'gpt':
        _llm_to_use = GPTModel(args.model_name)
    elif args.api == 'gemini':
        _llm_to_use = GeminiModel(args.model_name)
    else:
        raise ValueError("Invalid API type.")

    if os.path.exists(args.output_folder):
        print("Experiment folder already exists. Exiting...")
        sys.exit(0)
    else:
        os.makedirs(args.output_folder)
        print(f"Created folder: {args.output_folder}")

    for _i in range(num_of_repetitions):
        if args.selected_file is not None:
            _output_file = f'{args.output_folder}/{args.test_ds}_{_llm_to_use.get_model_name()}_{_instructions.__name__}_{_selected_file}_{_i}.json'
        else:
            _output_file = f'{args.output_folder}/{args.test_ds}_{_llm_to_use.get_model_name()}_{_instructions.__name__}_{_i}.json'

        start_time = time.time()
        main(test_folder=_test_folder, llm_to_use=_llm_to_use, instructions_func=_instructions,
             output_file=_output_file, selected_file=_selected_file)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
