import json
import re

import numpy as np
from transitions import Machine

from scripts.prompting_global.jup_utils import get_input_text
from scripts.prompting_timeline_fsm.prompts import extract_times, extract_timeline, extract_relations, \
    extract_times_missing_events, extract_missing_events_order, extract_timeline_order
from scripts.prompting_timeline_fsm.timeline_obj import Time, Event, Interval
from scripts.utils.omni_format_utils import filter_non_events

Event_Relations = ['BEFORE', 'AFTER', 'EQUAL', 'UNCERTAIN']


class TimelineSolverV2(object):
    states = ['init', 'init_timeline', 'missing_events_fulfilment', 'resolve_events_order', 'done']

    def __init__(self, agent, data):
        self.agent = agent
        self.timeline = Interval(Time.min, Time.max)
        self.machine = Machine(model=self, states=TimelineSolverV2.states, initial='init')
        self.machine.add_transition('start', 'init', 'init_timeline', after='initial_state')
        self.machine.add_transition('check_missing_events', 'init_timeline', 'missing_events_fulfilment', conditions='found_missing_events', after='resolve_missing_events')
        self.machine.add_transition('event_order', ['init_timeline','missing_events_fulfilment'], 'resolve_events_order', after='resolve_timeline_order')

        # self.machine.add_transition('resolve_intervals', ['missing_events_fulfilment', 'init_timeline'], 'resolve_intervals', unless='found_missing_events', after='extract_timeline_within_interval_events')
        # self.machine.add_transition('solve_disambiguation', '*', 'disambiguation', conditions='is_disambiguation', after='resolve_disambiguation')
        # self.machine.add_transition('check_missing_relations', 'disambiguation', 'missing_relations_fulfilment', conditions='found_missing_relations', after='extract_missing_relations')
        # self.machine.add_transition('done', '*', 'done', after='extract_timeline_between_interval_and_event_mix')

        self.data = data
        self.missing_events = None
        self.disambiguation = None
        self.graph = None
        self.event_ids = None

    def print_timeline(self, tab=0):
        self.timeline.print_timeline(tab=tab)

    def initial_state(self):
        text, _ = get_input_text(self.data)
        final_prompt = extract_times() + '\n' + text
        self.agent.add_message_from_instruct(final_prompt)
        response = self.agent.call_llm()

        self.extract_initial_timeline(self.extract_json(response))

    def found_missing_events(self):
        self.missing_events = self.validate_all_events_extracted(filter_non_events(self.data['allMentions']))
        return self.missing_events is not None and len(self.missing_events) > 0

    def resolve_missing_events(self):
        events = [event for event in filter_non_events(self.data['allMentions']) if int(event['m_id']) in self.missing_events]
        event_str = ", ".join([f"{event['tokens']}({event['m_id']})" for event in events])
        prompt = extract_times_missing_events(event_str)
        self.agent.add_message_from_instruct(prompt)
        response = self.agent.call_llm()
        events_ordered = self.extract_json(response)


    def resolve_timeline_order(self):
        prompt = extract_timeline_order()
        self.agent.add_message_from_instruct(prompt)
        response = self.agent.call_llm()
        missing_event_times = self.extract_json(response)
        self.extract_initial_timeline(missing_event_times)

    def extract_initial_timeline(self, response):
        print("Extracting initial timeline.")
        for key, value in response.items():
            text, m_id = Event.parse_key(key)
            start, end = Time.parse_value(value)

            if start[2] == 'XXXX' or end[2] == 'XXXX':
                continue

            fix_start = Time.fix_start(start)
            fix_end = Time.fix_end(end)
            event = Event(text, m_id, fix_start, fix_end)
            self.timeline.add_event(event)

        print("Initial timeline extracted.")
        self.timeline.print_timeline()
        print("-----------------------------------------")

    def validate_all_events_extracted(self, all_mentions):
        all_events = self.timeline.get_all_events()
        if len(all_events) != len(all_mentions):
            all_time_line_ids = [event.m_id for event in all_events]
            all_mention_ids = [int(mention['m_id']) for mention in all_mentions]
            missing_events = set(all_mention_ids) - set(all_time_line_ids)
            print(f"Missing events: {missing_events}")
            return missing_events
        return None

    def extract_timeline_within_interval_events(self):
        print("Extracting timeline between events within intervals.")
        stop = False
        stop_max_iter = 10
        loop_num = 0
        while not stop:
            if stop_max_iter == loop_num:
                print("Max iteration reached.")
                break

            unresolved_timeline = self.timeline.locate_unresolved_interval()
            if unresolved_timeline is not None:
                self.handle_interval_timeline(unresolved_timeline)
            else:
                stop = True
            loop_num += 1

        print(f"Timeline between events within intervals solved after {loop_num} iterations.")
        self.timeline.print_timeline()
        print("-----------------------------------------")

    def handle_interval_timeline(self, unresolved_timeline):
        events = unresolved_timeline.get_layer_events()
        event_str = ", ".join([f"{event.text}({event.m_id})" for event in events])
        prompt = extract_timeline(event_str)
        self.agent.add_message_from_instruct(prompt)
        response = self.agent.call_llm()
        data = self.extract_json(response)
        self.resolve_time_from_json(data, unresolved_timeline.timeline)

    def resolve_final_disambiguation(self):
        all_events = self.timeline.get_all_events()
        starting_date_group = {}
        for event1 in all_events:
            for event2 in all_events:
                if event1 != event2:
                    if event1.start == event2.start and (event1.order == -1 or event2.order == -1):
                        if f'{event1.start}' not in starting_date_group:
                            starting_date_group[f'{event1.start}'] = set()
                        starting_date_group[f'{event1.start}'].add(event1)
                        starting_date_group[f'{event1.start}'].add(event2)
        self.disambiguation = starting_date_group

    def is_disambiguation(self):
        self.resolve_final_disambiguation()
        return len(self.disambiguation) > 0

    def resolve_disambiguation(self):
        results = dict()
        for start_date, events in self.disambiguation.items():
            print(f"Resolve the following events with the same starting date: {start_date}")
            event_str = ", ".join([f"{event.text}({event.m_id})" for event in events])
            prompt = extract_timeline(event_str)
            self.agent.add_message_from_instruct(prompt)
            response = self.agent.call_llm()
            results[start_date] = self.extract_json(response)

            # insert_final_pairs_into_graph(graph, event_ids, result_dict)
        for start_date, data_json in results.items():
            print(f"Resolved pairs for date: {start_date}")
            self.resolve_time_from_json(data_json, list(self.disambiguation[start_date]))

        print("Disambiguation resolved.")
        self.timeline.print_timeline()
        print("-----------------------------------------")

    def found_missing_relations(self):
        if self.disambiguation is not None:
            for start_date, events in self.disambiguation.items():
                for event in events:
                    if event.order == -1:
                        return True
            self.disambiguation = None
        return False

    def extract_missing_relations(self):
        missed_ordered_events = []
        for start_date, events in self.disambiguation.items():
            for event in events:
                if event.order == -1:
                    missed_ordered_events.append(event)

        if len(missed_ordered_events) != 0:
            event_str = ", ".join([f"{event.text}({event.m_id})" for event in missed_ordered_events])
            prompt = extract_missing_events_order(event_str)
            self.agent.add_message_from_instruct(prompt)
            response = self.agent.call_llm()
            data = self.extract_json(response)
            self.resolve_time_from_json(data, missed_ordered_events)

    @staticmethod
    def resolve_time_from_json(data, unresolved_timeline):
        for event_id, order in data.items():
            event, _id = Event.parse_key(event_id)
            for inter_event in unresolved_timeline:
                if inter_event.m_id == _id:
                    inter_event.order = order

    def extract_timeline_between_interval_and_event_mix(self):
        print("Building the graph and preparing final disambiguation.")
        all_events = self.timeline.get_all_events()
        all_events_ids = {event.m_id: i for i, event in enumerate(all_events)}
        graph = np.full((len(all_events_ids), len(all_events_ids)), -1)
        starting_date_group = {}
        for event1 in all_events:
            for event2 in all_events:
                if event1 != event2:
                    if event1.order == 'X' or event2.order == 'X':
                        graph[all_events_ids[event1.m_id]][all_events_ids[event2.m_id]] = Event_Relations.index('UNCERTAIN')
                        graph[all_events_ids[event2.m_id]][all_events_ids[event1.m_id]] = Event_Relations.index('UNCERTAIN')
                    elif event1.start < event2.start:
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

        print("Found the following events with the same starting date:")
        for date, events in starting_date_group.items():
            print(f"Date: {date}, Events: {', '.join([event.text for event in events])}")

        self.graph = graph
        self.event_ids = all_events_ids

    @staticmethod
    def extract_json(response):
        match = re.search(r'\{.*}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)  # Convert to dictionary
            except json.JSONDecodeError:
                return None  # Invalid JSON
        return None

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
