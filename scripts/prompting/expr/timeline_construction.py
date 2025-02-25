import json
import re
from copy import copy
from datetime import datetime


class IComponent(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @staticmethod
    def fix_start(time):
        ret_time = copy(time)
        if time[0] == 'XX':
            ret_time[0] = '01'
        if time[1] == 'XX':
            ret_time[1] = '01'
        if time[2] == 'XXXX' or time[2] == 'YYYY':
            ret_time[2] = '0001'
        return datetime(int(ret_time[2]), int(ret_time[1]), int(ret_time[0]), 0, 0)

    @staticmethod
    def fix_end(time):
        ret_time = copy(time)
        if time[0] == 'XX':
            ret_time[0] = '28'
        if time[1] == 'XX':
            ret_time[1] = '12'
        if time[2] == 'XXXX' or time[2] == 'YYYY':
            ret_time[2] = '5000'
        return datetime(int(ret_time[2]), int(ret_time[1]), int(ret_time[0]), 0, 0)

    def is_encapsulating(self, start, end):
        if self.start <= start and self.end >= end:
            return self

        return None


class Interval(IComponent):
    def __init__(self, start, end):
        super().__init__(start, end)
        self.timeline = list()

    def set_interval(self, interval):
        pass

    def add_first(self, event_or_interval):
        self.timeline.append(event_or_interval)
        event_or_interval.set_interval(self)

    def get_interval(self, event):
        ret_interval = None
        for interval in self.timeline:
            if isinstance(interval, Interval):
                if interval.is_encapsulating(event.start, event.end):
                    if ret_interval is None:
                        ret_interval = interval
                    elif ret_interval.start > interval.start or ret_interval.end < interval.end:
                        ret_interval = interval

        return ret_interval

    def get_event(self, event):
        ret_event = None
        for loop_event in self.timeline:
            if isinstance(loop_event, Event):
                if loop_event.is_encapsulating(event.start, event.end):
                    if ret_event is None:
                        ret_event = loop_event
                    elif ret_event.start < loop_event.start or ret_event.end > loop_event.end:
                        ret_event = loop_event

        return ret_event

    def add_event(self, event):
        is_added = False
        interval = self.get_interval(event)
        inner_event = self.get_event(event)
        if interval:
            is_added = interval.add_event(event)

        if not is_added:
            if inner_event:
                inner_event_interval = inner_event.interval
                if inner_event_interval is not None:
                    if inner_event_interval.start == inner_event.start and inner_event_interval.end == inner_event.end:
                        inner_event_interval.add_first(event)
                        return True
                    else:
                        new_interval = Interval(inner_event.start, inner_event.end)
                        new_interval.add_first(inner_event)
                        new_interval.add_first(event)
                        self.timeline.remove(inner_event)
                        self.timeline.append(new_interval)
                        return True

            # Didn't find a container for the event, seeing if the event can be a container before adding it
            self.add_with_encapsulation_check(event)
            is_added = True
        return is_added

    def add_with_encapsulation_check(self, event):
        new_interval = Interval(event.start, event.end)
        is_empty_interval = True
        for idx in range(len(self.timeline) - 1, -1, -1):
            interval_event = self.timeline[idx]
            if event.is_encapsulating(interval_event.start, interval_event.end):
                new_interval.add_first(interval_event)
                new_interval.add_first(event)
                self.timeline.remove(interval_event)
                self.timeline.append(new_interval)
                is_empty_interval = False
        if is_empty_interval:
            self.add_first(event)

    def add_events(self, events):
        self.timeline.extend(events)

    def add_interval(self, interval):
        self.timeline.append(interval)

    def __str__(self):
        return f"{self.start.strftime('%d:%m:%Y')}--{self.end.strftime('%d:%m:%Y')}"


class Event(IComponent):
    def __init__(self, text, m_id, start, end):
        super().__init__(start, end)
        self.text = text
        self.m_id = m_id
        self.interval = None

    def set_interval(self, interval):
        self.interval = interval

    def __str__(self):
        return f"{self.text}({self.m_id}) # {self.start.strftime('%d:%m:%Y')}--{self.end.strftime('%d:%m:%Y')}"


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


def main(in_file):
    data = json.load(open(in_file))
    timeline = Interval(datetime.min, datetime.max)
    # unknown_dates = list()
    for key, value in data.items():
        text, m_id = parse_key(key)
        start, end = parse_value(value)
        fix_start = IComponent.fix_start(start)
        fix_end = IComponent.fix_end(end)
        event = Event(text, m_id, fix_start, fix_end)
        timeline.add_event(event)

    print(timeline)


if __name__ == "__main__":
    _in_file = "data/my_data/expr/temporal_expr/155d4_event_times3.json"
    main(_in_file)
