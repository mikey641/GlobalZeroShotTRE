from scripts.eval.dataset.utils import parse_DOT
from scripts.eval.model.compute_metrics import calculate
from scripts.eval.shared.eval_obj import EvalObj
from scripts.utils.io_utils import open_input_file


def load_in_dot(predictions):
    all_predictions = dict()
    for file in predictions.keys():
        predicted_graph = parse_DOT(predictions[file]['target'])
        if predicted_graph is None:
            print(f'Error: {file}')

        all_predictions[file] = predicted_graph
    return all_predictions


def convert_to_evalobj(all_predictions):
    eval_objs = dict()
    for doc_id in all_predictions:
        gen_edge_list = all_predictions[doc_id]
        gen_eval_obj = EvalObj(doc_id, gen_edge_list)
        eval_objs[doc_id] = gen_eval_obj
    return eval_objs


def get_all_edges(eval_objs):
    edges = dict()
    for doc_id, edge_list in eval_objs.items():
        all_doc_edges = dict()
        for edge in edge_list.edge_set:
            all_doc_edges[f'{edge[0]}--{edge[2]}'] = {'before': 0, 'after': 0, 'equal': 0, 'vague': 0}
        edges[doc_id] = all_doc_edges
    return edges


def calc_prediction_voting(pred_data, voting_data):
    for doc_id, pred_edge_list in pred_data.items():
        count_votes = 0
        for edge in pred_edge_list.edge_set:
            edge_key = f'{edge[0]}--{edge[2]}'
            voting_data[doc_id][edge_key][edge[1]] += 1
            count_votes += 1


if __name__ == "__main__":
    _prediction_file1 = 'data/my_data/predictions/eventfull/outputs/eventfull_gpt4o_1exmp_rand_29_task_description_v2.json'
    _prediction_file2 = 'data/my_data/predictions/eventfull/outputs/eventfull_gpt4o_1exmp_rand_28_task_description_v2.json'
    _prediction_file3 = 'data/my_data/predictions/eventfull/outputs/eventfull_gpt4o_1exmp_rand_25_task_description_v2.json'
    _prediction_file4 = 'data/my_data/predictions/eventfull/outputs/eventfull_gpt4o_1exmp_rand_27_task_description_v2.json'
    _prediction_file5 = 'data/my_data/predictions/eventfull/outputs/eventfull_gpt4o_1exmp_rand_26_task_description_v2.json'
    _gold_file = 'data/DOT_format/EventFull_test_dot.json'

    _pred1_data = convert_to_evalobj(load_in_dot(open_input_file(_prediction_file1)))
    _pred2_data = convert_to_evalobj(load_in_dot(open_input_file(_prediction_file2)))
    _pred3_data = convert_to_evalobj(load_in_dot(open_input_file(_prediction_file3)))
    _pred4_data = convert_to_evalobj(load_in_dot(open_input_file(_prediction_file4)))
    _pred5_data = convert_to_evalobj(load_in_dot(open_input_file(_prediction_file5)))
    _gold_data = convert_to_evalobj(load_in_dot(open_input_file(_gold_file)))

    _all_edges_voting = get_all_edges(_gold_data)
    calc_prediction_voting(_pred1_data, _all_edges_voting)
    calc_prediction_voting(_pred2_data, _all_edges_voting)
    calc_prediction_voting(_pred3_data, _all_edges_voting)
    calc_prediction_voting(_pred4_data, _all_edges_voting)
    calc_prediction_voting(_pred5_data, _all_edges_voting)

    # find the majority vote for each class in each edge (consider ties as 'vague')
    final_majority_voting = dict()
    for doc_id, doc_edges in _all_edges_voting.items():
        final_majority_voting[doc_id] = list()
        for edge_key, edge_votes in doc_edges.items():
            edge_nodes = edge_key.split('--')
            max_vote = max(edge_votes.values())
            if list(edge_votes.values()).count(max_vote) > 1:
                final_majority_voting[doc_id].append((edge_nodes[0], 'vague', edge_nodes[1]))
            else:
                final_majority_voting[doc_id].append((edge_nodes[0], max(edge_votes, key=edge_votes.get), edge_nodes[1]))

    _all_predictions = dict()
    _orig_gold = open_input_file(_gold_file)
    for _file in final_majority_voting.keys():
        _gold_graph = parse_DOT(_orig_gold[_file]['target'])
        _all_predictions[_file] = {"generated": final_majority_voting[_file], "gold": _gold_graph}

    calculate(_all_predictions)
    print('Done!')
