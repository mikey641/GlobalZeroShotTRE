import json
import traceback

from scripts.eval.shared.evaluation import evaluation
from scripts.prompting.fsm_experiment.agent_obj import GPTAgent
from scripts.prompting.fsm_experiment.timeline_solver_v1 import Event_Relations, TimelineSolverV1
from scripts.prompting.fsm_experiment.timeline_solver_v2 import TimelineSolverV2
from scripts.utils.io_utils import open_input_file


def from_graph_to_pairs(graph, event_ids):
    reverse_event_id = {v: k for k, v in event_ids.items()}
    pairs = {}
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != -1:
                pairs[f'{reverse_event_id[i]}#{reverse_event_id[j]}'] = Event_Relations[graph[i][j]].lower()

    return pairs


# states = ['init', 'init_timeline', 'check_missing_events', 'resolve_intervals', 'disambiguation', 'done']
def main_v1(timeline_solver):
    print("Initial state: ", timeline_solver.state)
    timeline_solver.start()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.check_missing_events()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.resolve_intervals()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.solve_disambiguation()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.check_missing_relations()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.done()
    print("Transitioning to state: ", timeline_solver.state)

    pred_pairs = from_graph_to_pairs(timeline_solver.graph, timeline_solver.event_ids)
    gold_pairs = {f"{pair['_firstId']}#{pair['_secondId']}": pair['_relation'] for pair in data['allPairs']}

    golds = []
    preds = []
    for key, value in gold_pairs.items():
        assert key in pred_pairs, f"Key {key} not found in prediction pairs."
        golds.append(value)
        preds.append(pred_pairs[key])
        if value != pred_pairs[key]:
            print(f"found inconsistent pair-{key}: Gold-{value}, Pred-{pred_pairs[key]}")

    return golds, preds


def main_v2(timeline_solver):
    timeline_solver.start()
    print("Initial state: ", timeline_solver.state)
    timeline_solver.check_missing_events()
    print("Transitioning to state: ", timeline_solver.state)
    timeline_solver.event_order()
    print("Transitioning to state: ", timeline_solver.state)


if __name__ == "__main__":
    _doc_num = '6'
    _in_initial_doc = f"data/OmniTemp/test/{_doc_num}_final.json"
    data = open_input_file(f'{_in_initial_doc}')
    _agent = GPTAgent(gpt4o_mini)
    # _agent = GPTAgentSimulator(json.load(open(f"data/my_data/expr/sim/test/{_doc_num}_sim_v1.json")))
    _timeline_solver = TimelineSolverV1(_agent, data)
    try:
        if type(_timeline_solver) is TimelineSolverV1:
            _golds, _preds = main_v1(_timeline_solver)
        elif type(_timeline_solver) is TimelineSolverV2:
            _golds, _preds = main_v2(_timeline_solver)
        else:
            raise TypeError('Invalid timeline solver type')
    except Exception as e:
        traceback.print_exc()
        _golds, _preds = None, None

    print("Agent Messages: " + json.dumps(_agent.get_messages(), indent=4))
    print("-----------------------------------------")
    _timeline_solver.print_timeline()
    print("-----------------------------------------")

    if _golds and _preds:
        evaluation(_golds, _preds)
