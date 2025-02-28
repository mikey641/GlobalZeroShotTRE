import json
import re
import traceback
from itertools import combinations

import numpy as np

from scripts.prompting_global.jup_utils import get_input_text
from scripts.prompting_timeline_fsm.agent_obj import GPTAgentSimulator, GPTAgent
from scripts.prompting_timeline_fsm.eval import evaluation
from scripts.prompting_timeline_fsm.prompts import extract_times, extract_timeline, extract_relations
from scripts.prompting_timeline_fsm.timeline_obj import Time, Event, Interval
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import gpt4o_mini


Event_Relations = ['BEFORE', 'AFTER', 'EQUAL', 'UNCERTAIN']


def get_reverse_relation(relation):
    if relation == 'BEFORE':
        return 'AFTER'
    elif relation == 'AFTER':
        return 'BEFORE'
    return relation


def insert_into_timeline(timeline, date, event, m_id):
    if date[2] in timeline:
        year = timeline[date[2]]
        if date[1] in year:
            month = year[date[1]]
            if date[0] in month:
                month[date[0]].append((event, m_id))
            else:
                month[date[0]] = [(event, m_id)]
        else:
            year[date[1]] = {date[0]: [(event, m_id)]}
    else:
        timeline[date[2]] = {date[1]: {date[0]: [(event, m_id)]}}


def extract_initial_timeline(data, timeline):
    for key, value in data.items():
        text, m_id = Event.parse_key(key)
        start, end = Time.parse_value(value)
        fix_start = Time.fix_start(start)
        fix_end = Time.fix_end(end)
        event = Event(text, m_id, fix_start, fix_end)
        timeline.add_event(event)

    return timeline


def extract_timeline_within_interval_events(agent, timeline):
    stop = False
    stop_max_iter = 10
    loop_num = 0
    while not stop:
        if stop_max_iter == loop_num:
            print("Max iteration reached.")
            break

        unresolved_timeline = timeline.locate_unresolved_interval()
        if unresolved_timeline is not None:
            handle_interval_timeline(agent, unresolved_timeline)
        else:
            stop = True
        loop_num += 1

    print(f"Timeline between events within intervals solved after {loop_num} iterations.")
    return timeline


def extract_timeline_between_interval_and_event_mix(timeline):
    all_events = timeline.get_all_events()
    all_events_ids = {event.m_id: i for i, event in enumerate(all_events)}
    graph = np.full((len(all_events_ids), len(all_events_ids)), -1)
    starting_date_group = {}
    for event1 in all_events:
        for event2 in all_events:
            if event1 != event2:
                if event1.start < event2.start:
                    graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('BEFORE')
                    graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('AFTER')
                elif event1.start > event2.start:
                    graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('AFTER')
                    graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('BEFORE')
                elif event1.start == event2.start:
                    if event1.order != -1 and event2.order != -1:
                        if event1.order == event2.order:
                            graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('EQUAL')
                            graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('EQUAL')
                        elif event1.order < event2.order:
                            graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('BEFORE')
                            graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('AFTER')
                        elif event1.order > event2.order:
                            graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('AFTER')
                            graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('BEFORE')
                    else:
                        if f'{event1.start}' not in starting_date_group:
                            starting_date_group[f'{event1.start}'] = set()
                        starting_date_group[f'{event1.start}'].add(event1)
                        starting_date_group[f'{event1.start}'].add(event2)

    return graph, all_events_ids, starting_date_group


def insert_final_pairs_into_graph(graph, all_events_ids, pairs_dict_json):
    for start_date, pairs_list in pairs_dict_json.items():
        for pair in pairs_list:
            source_key = pair['source']
            source_text, source_id = Event.parse_key(source_key)
            target_key = pair['target']
            target_text, target_id = Event.parse_key(target_key)
            relation = pair['relation']
            graph[all_events_ids[source_id]][all_events_ids[target_id]] = Event_Relations.index(relation.upper())
            graph[all_events_ids[target_id]][all_events_ids[source_id]] = Event_Relations.index(get_reverse_relation(relation).upper())


def handle_interval_timeline(agent, unresolved_timeline):
    events = unresolved_timeline.get_layer_events()
    event_str = ", ".join([f"{event.text}({event.m_id})" for event in events])
    prompt = extract_timeline(event_str)
    agent.add_message_from_instruct(prompt)
    response = agent.call_llm()
    data = extract_json(response)

    for event_id, order in data.items():
        event, _id = Event.parse_key(event_id)
        for inter_event in unresolved_timeline.timeline:
            if inter_event.m_id == _id:
                inter_event.order = order


def initial_state(agent, data):
    text, _ = get_input_text(data)
    final_prompt = extract_times() + '\n' + text
    agent.add_message_from_instruct(final_prompt)
    response = agent.call_llm()

    return extract_json(response)


def extract_json(response):
    match = re.search(r'\{.*}', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)  # Convert to dictionary
        except json.JSONDecodeError:
            return None  # Invalid JSON
    return None


def extract_json_list(response):
    match = re.search(r'\[.*]', response, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)  # Convert to dictionary
        except json.JSONDecodeError:
            return None  # Invalid JSON
    return None


def from_graph_to_pairs(graph, event_ids):
    reverse_event_id = {v: k for k, v in event_ids.items()}
    pairs = {}
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] != -1:
                pairs[f'{reverse_event_id[i]}#{reverse_event_id[j]}'] = Event_Relations[graph[i][j]].lower()

    return pairs


def final_pairs_to_resolve(agent, starting_date_groups):
    results = dict()
    for start_date, events in starting_date_groups.items():
        print(f"Resolve the following events with the same starting date: {start_date}")
        event_str = ", ".join([f"{event.text}({event.m_id})" for event in events])
        all_pairs = list(combinations(events, 2))
        pairs_str = "\n".join([f"{pair[0].text}({pair[0].m_id}) - - {pair[1].text}({pair[1].m_id})" for pair in all_pairs])
        prompt = extract_relations(event_str, pairs_str)
        agent.add_message_from_instruct(prompt)
        response = agent.call_llm()
        results[start_date] = extract_json_list(response)

    return results


def main(agent, timeline, doc_file):
    data = open_input_file(f'{doc_file}')
    event_times = initial_state(agent, data)
    timeline = extract_initial_timeline(event_times, timeline)
    timeline = extract_timeline_within_interval_events(agent, timeline)
    graph, event_ids, starting_date_group = extract_timeline_between_interval_and_event_mix(timeline)

    if len(starting_date_group) > 0:
        result_dict = final_pairs_to_resolve(agent, starting_date_group)
        insert_final_pairs_into_graph(graph, event_ids, result_dict)

    pred_pairs = from_graph_to_pairs(graph, event_ids)
    gold_pairs = {f"{pair['_firstId']}#{pair['_secondId']}": pair['_relation'] for pair in data['allPairs']}

    golds = []
    preds = []
    for key, value in gold_pairs.items():
        assert key in pred_pairs, f"Key {key} not found in prediction pairs."
        golds.append(value)
        preds.append(pred_pairs[key])
        if value != pred_pairs[key]:
            print(f"found inconsistent pair-{key}: Gold-{value}, Pred-{pred_pairs[key]}")

    evaluation(golds, preds)


if __name__ == "__main__":
    _in_initial_doc = "data/OmniTemp/train/30_final.json"
    _agent = GPTAgent(gpt4o_mini)
    # _agent = GPTAgentSimulator()
    _timeline = Interval(Time.min, Time.max)
    try:
        main(_agent, _timeline, _in_initial_doc)
    except Exception as e:
        traceback.print_exc()

    print(json.dumps(_agent.get_messages(), indent=4))
    print("-----------------------------------------")
    _timeline.print_timeline()
