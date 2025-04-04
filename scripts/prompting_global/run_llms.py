import json
import os
import time

import argparse
from tqdm import tqdm

from scripts.eval.dataset.utils import find_ment_by_id
from scripts.prompting_global.jup_utils import get_input_text, prepare_instructions, get_all_pairs
from scripts.prompting_global.prompts_gpt import task_description_4res_only_timeline, task_description_4res_only_global, \
    task_description_6res_only_global, task_description_6res_only_timeline, task_description_6rels
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import TogetherModel, GPTModel, GeminiModel


def get_example_matrix(pairs, all_ment_ids):
    example_matrix = [[0 for _ in range(len(all_ment_ids))] for _ in range(len(all_ment_ids))]
    for pair in pairs:
        first_id = pair['_firstId']
        second_id = pair['_secondId']
        relation = pair['_relation']
        if '/' in relation:
            split_rel = relation.split('/')
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = split_rel[0]
        else:
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = relation

    # pretty print matrix
    print("Expected matrix:")
    for row in example_matrix:
        print(row)

    return example_matrix


def get_prompt_pairs(mentions, gold_pairs):
    prompt_pairs = list()
    example_with_rels = list()
    for pair in gold_pairs:
        first_ment = find_ment_by_id(mentions, pair['_firstId'])
        second_ment = find_ment_by_id(mentions, pair['_secondId'])
        first_pair = f'{first_ment["tokens"]}({first_ment["m_id"]})'
        second_pair = f'{second_ment["tokens"]}({second_ment["m_id"]})'
        prompt_pairs.append(f'{first_pair}/{second_pair}')
        example_with_rels.append(f'{first_pair}/{second_pair}={pair["_relation"]}')
    return prompt_pairs, example_with_rels


def main(test_folder, llm_to_use, instructions_func, output_file,
         number_of_pred, selected_file, gen_pairs=True):
    # load json file
    predictions = dict()
    final_instructions = instructions_func(None)
    count = 0
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        if count == number_of_pred:
            break

        if selected_file is not None and file1 != selected_file:
            continue

        print("Processing file:", file1)

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text, all_pairs = get_input_text(data)

        task_desc = final_instructions + '\n' + text
        if gen_pairs:
            all_mentions = data['allMentions']
            # all_ment_ids = {m['m_id']: f"{m['tokens']}({m['m_id']})" for m in all_mentions}
            all_ment_ids = {m['m_id']: m for m in all_mentions}
            example_matrix = get_all_pairs(all_pairs, all_ment_ids, reduction=-1)
            task_desc += '\nPairs require classification:\n' + '\n'.join(example_matrix)

        try:
            response = llm_to_use.run_model(task_desc)
            predictions[file1] = {"target": response}
        except Exception as e:
            predictions[file1] = {"target": "Generation Failed"}

    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run llm on test data")
    parser.add_argument("--test_db", help="The test database name (nt, matres, omni, tbd)", type=str, required=True)
    parser.add_argument("--num_or_pred", help="The number of predictions to make (default all docs)", type=int)
    parser.add_argument("--instruct", help="Should be global/timeline", type=str, required=True)
    parser.add_argument("--api", help="Should be together/gpt/gemini", type=str, required=True)
    parser.add_argument("--model_name", help="the model to use", type=str, required=True)
    parser.add_argument("--repeat", help="Times to repeat the experiment", type=int, required=True)
    parser.add_argument("--selected_file", help="To only run a single specific file", type=str, required=False)

    args = parser.parse_args()
    print(vars(args))

    _num_of_pred = args.num_or_pred
    num_of_repetitions = args.repeat
    _selected_file = args.selected_file

    if args.test_db == "nt":
        _test_folder = 'data/NarrativeTime_A1/converted_no_overlap/test_18ment'
    elif args.test_db == "matres":
        _test_folder = 'data/MATRES/in_my_format_all_pairs/test'
    elif args.test_db == "omni":
        _test_folder = 'data/OmniTemp/test'
    elif args.test_db == "tbd":
        _test_folder = 'data/TBDense/test'
    else:
        raise ValueError("Invalid test database name.")

    if args.instruct == 'global' and args.test_db in ["nt", "tb"]:
        _instructions = task_description_6res_only_global
    elif args.instruct == 'global' and args.test_db in ["matres", "omni"]:
        _instructions = task_description_4res_only_global
    elif args.instruct == 'timeline' and args.test_db in ["nt", "tb"]:
        _instructions = task_description_6res_only_timeline
    elif args.instruct == 'timeline' and args.test_db in ["matres", "omni"]:
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

    for i in range(num_of_repetitions):
        if args.selected_file is not None:
            _output_file = f'data/my_data/predictions/new_expr/{args.test_db}_{_llm_to_use.get_model_name()}_{_instructions.__name__}_{_selected_file}.json'
        else:
            _output_file = f'data/my_data/predictions/new_expr/{args.test_db}_{_llm_to_use.get_model_name()}_{_instructions.__name__}_{i}.json'

        start_time = time.time()
        main(test_folder=_test_folder, llm_to_use=_llm_to_use, instructions_func=_instructions,
             output_file=_output_file, selected_file=_selected_file, number_of_pred=_num_of_pred)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
