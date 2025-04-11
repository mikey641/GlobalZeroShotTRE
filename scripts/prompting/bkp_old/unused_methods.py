from scripts.eval.dataset.utils import find_ment_by_id


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
