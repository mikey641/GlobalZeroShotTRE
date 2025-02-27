import json
import os
import time

from tqdm import tqdm

from scripts.eval.dataset.utils import find_ment_by_id
from scripts.prompting_global.jup_utils import get_input_text, prepare_instructions, get_all_pairs
from scripts.prompting_global.prompts import task_description_4res_only_timeline, task_description_4res_only_global, \
    task_description_6res_only_global, task_description_6res_only_timeline
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import gpt4o

gemini_pro_model = None
gemini_flash_model = None


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


def main(test_folder, train_folder, dot_train_data, llm_to_use, instructions_func, output_file,
         number_of_pred, prompt_examples, selected_file, reduction=-1.0, gen_pairs=False):
    # load json file
    predictions = dict()
    # examples = prepare_instructions(train_folder, dot_train_data, prompt_examples, selected_file, reduction)
    final_instructions = instructions_func(None)
    count = 0
    for i, file1 in enumerate(tqdm(os.listdir(test_folder))):
        if count == number_of_pred:
            break

        if selected_file is not None and file1 != selected_file:
            continue

        count += 1
        data = open_input_file(f'{test_folder}/{file1}')
        text, all_pairs = get_input_text(data)

        task_desc = final_instructions + '\n' + text
        if gen_pairs:
            all_mentions = data['allMentions']
            # all_ment_ids = {m['m_id']: f"{m['tokens']}({m['m_id']})" for m in all_mentions}
            all_ment_ids = {m['m_id']: m for m in all_mentions}
            example_matrix = get_all_pairs(all_pairs, all_ment_ids, reduction)
            task_desc += '\nPairs require classification:\n' + '\n'.join(example_matrix)

        try:
            response = llm_to_use(task_desc)
            predictions[file1] = {"target": response}
        except Exception as e:
            predictions[file1] = {"target": "Generation Failed"}

    with open(output_file, 'w') as file:
        json.dump(predictions, file, indent=4)


if __name__ == "__main__":
    example_db = 'OmniTemp'
    test_db = 'OmniTemp'
    num_of_pred = -1
    num_of_prompt_examples = 1
    _reduction = -1
    # APW19980213.1380.json
    _selected_file = '21_final.json'
    _instructions = task_description_4res_only_timeline
    _llm_to_use = gpt4o
    # _test_folder = 'data/MATRES/in_my_format_all_pairs/test'
    _test_folder = 'data/OmniTemp/train'
    # _dot_test_data = open_input_file('data/DOT_format/MATRES_test_dot.json')
    # _test_folder = 'data/OmniTemp/test'

    _train_folder = 'data/MATRES/in_my_format/train'
    _dot_train_data = open_input_file('data/DOT_format/MATRES_train_dot.json')
    # _train_folder = 'data/OmniTemp/train'
    # _dot_train_data = open_input_file('data/DOT_format/EventFull_dev_dot.json')

    # _output_file = f'data/my_data/predictions/{test_db}/{example_db}_{_llm_to_use.__name__}_{num_of_pred}pred_{num_of_prompt_examples}exmples_{_instructions.__name__}.json'
    _output_file = f'data/my_data/predictions/{test_db}/test_{example_db}_{_llm_to_use.__name__}_{num_of_pred}pred_{num_of_prompt_examples}exmp_{_instructions.__name__}.json'

    start_time = time.time()
    main(test_folder=_test_folder, train_folder=_train_folder,
         dot_train_data=_dot_train_data, llm_to_use=_llm_to_use, instructions_func=_instructions,
         output_file=_output_file, number_of_pred=num_of_pred, prompt_examples=num_of_prompt_examples,
         selected_file=_selected_file, reduction=_reduction, gen_pairs=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
