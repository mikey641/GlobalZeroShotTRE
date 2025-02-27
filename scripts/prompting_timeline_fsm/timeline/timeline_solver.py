import json
import re

from scripts.prompting_global.jup_utils import get_input_text
from scripts.prompting_timeline_fsm.prompts import extract_times, extract_timeline
from scripts.prompting_timeline_fsm.timeline.agent_obj import GPTAgent, GPTAgentSimulator
from scripts.prompting_timeline_fsm.timeline.timeline_obj import Interval, Time, Event
from scripts.utils.io_utils import open_input_file
from scripts.utils.llms_definitions import gpt4o_mini


def validate_date(date):
    if date[0] == 'XX' and date[1] == 'XX' and (date[2] == 'XXXX' or date[2] == 'YYYY'):
        return False
    return True


def parse_value(value):
    start_end = value.split('--')
    start = start_end[0].split(':')
    end = start_end[1].split(':')
    return start, end


def parse_key(key):
    match = re.match(r'([a-zA-Z]+)\((\d+)\)', key)
    if match:
        event, m_id = match.groups()  # Extracts the text and the number
        return event, int(m_id)  # Convert B to an integer
    else:
        raise ValueError(f"Invalid format: {key}")


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
        text, m_id = parse_key(key)
        start, end = parse_value(value)
        fix_start = Time.fix_start(start)
        fix_end = Time.fix_end(end)
        event = Event(text, m_id, fix_start, fix_end)
        timeline.add_event(event)

    return timeline


def extract_timeline_between_interval_event(agent, timeline):
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


def extract_timeline_between_interval_and_event(timeline):
    stop = False
    stop_max_iter = 10
    loop_num = 0
    while not stop:
        if stop_max_iter == loop_num:
            print("Max iteration reached.")
            break

        unresolved_timeline = timeline.locate_unresolved_interval_events_mix()
        if unresolved_timeline is not None:
            handle_mix_interval_timeline(unresolved_timeline)
        else:
            stop = True
        loop_num += 1

    print(f"Timeline between events within intervals solved after {loop_num} iterations.")
    return timeline


def handle_mix_interval_timeline(unresolved_timeline):
    pass


def handle_interval_timeline(agent, unresolved_timeline):
    events = unresolved_timeline.get_all_events()
    event_str = ", ".join([f"{event.text}({event.m_id})" for event in events])
    prompt = extract_timeline(event_str)
    agent.add_message_from_instruct(prompt)
    response = agent.call_llm()
    data = extract_json(response)

    for event_id, order in data.items():
        event, _id = parse_key(event_id)
        for inter_event in unresolved_timeline.timeline:
            if inter_event.m_id == _id:
                inter_event.start.order = order
                inter_event.end.order = order


def initial_state(agent, doc_file):
    data = open_input_file(f'{doc_file}')
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


def main(agent, timeline, doc_file):
    event_times = initial_state(agent, doc_file)
    timeline = extract_initial_timeline(event_times, timeline)
    timeline = extract_timeline_between_interval_event(agent, timeline)
    timeline = extract_timeline_between_interval_and_event(timeline)
    return timeline


if __name__ == "__main__":
    _in_initial_doc = "data/OmniTemp/train/30_final.json"
    # _agent = GPTAgent(gpt4o_mini)
    _agent = GPTAgentSimulator()
    _timeline = Interval(Time.min, Time.max)
    try:
        main(_agent, _timeline, _in_initial_doc)
    except Exception as e:
        print(repr(e))

    print(json.dumps(_agent.get_messages(), indent=4))
    print("-----------------------------------------")
    _timeline.print_timeline()
