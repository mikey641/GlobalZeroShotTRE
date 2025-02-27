import json
import re

from scripts.prompting_global.jup_utils import get_input_text
from scripts.prompting_timeline_fsm.timeline.timeline_obj import Interval, Time, Event
from scripts.utils.io_utils import open_input_file


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


def extract_initial_timeline(in_file):
    data = json.load(open(in_file))
    timeline = Interval(Time.min, Time.max)
    # unknown_dates = list()
    for key, value in data.items():
        text, m_id = parse_key(key)
        start, end = parse_value(value)
        fix_start = Time.fix_start(start)
        fix_end = Time.fix_end(end)
        event = Event(text, m_id, fix_start, fix_end)
        timeline.add_event(event)

    return timeline


def extract_timeline_between_interval_and_event(in_file):
    pass

def extract_timeline_between_events(dot):
    pass


def read_and_set_timeline(in_file, unresolved_timeline):
    # open and load json file
    with open(in_file) as json_file:
        data = json.load(json_file)

    for event_id, order in data.items():
        event, _id = parse_key(event_id)
        for inter_event in unresolved_timeline.timeline:
            if inter_event.m_id == _id:
                inter_event.start.order = order
                inter_event.end.order = order


def find_unresolved_timeline(timeline):
    return timeline.locate_unresolved_interval()


def initial_state(doc_file):
    data = open_input_file(f'{doc_file}')
    text, _ = get_input_text(data)


def main(in_file, _in_timeline1):
    timeline = extract_initial_timeline(in_file)
    stop = False
    stop_max_iter = 10
    loop_num = 0
    while not stop:
        if stop_max_iter == loop_num:
            break

        unresolved_timeline = find_unresolved_timeline(timeline)
        if unresolved_timeline is not None:
            read_and_set_timeline(_in_timeline1, unresolved_timeline)
        else:
            stop = True
        loop_num += 1

    timeline.print_timeline()


if __name__ == "__main__":
    _in_initial_doc = "data/OmniTemp/train/30_final.json"
    _in_time_file = "data/my_data/expr/time_expr/155d4_event_times3.json"
    _in_timeline1 = "data/my_data/expr/timelines/155d4_event_timeline1.json"
    main(_in_time_file, _in_timeline1)
