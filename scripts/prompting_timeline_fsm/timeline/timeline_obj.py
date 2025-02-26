from copy import copy


class Time(object):
    def __init__(self, year, month, day, order=None):
        self.year = year
        self.month = month
        self.day = day
        self.order = order

    @staticmethod
    def min():
        return Time(1, 1, 1, 0)

    @staticmethod
    def max():
        return Time(9999, 12, 31, 0)

    @staticmethod
    def fix_start(time):
        ret_time = copy(time)
        if time[0] == 'XX':
            ret_time[0] = 1
        if time[1] == 'XX':
            ret_time[1] = 1
        if time[2] == 'XXXX' or time[2] == 'YYYY':
            ret_time[2] = 1

        return Time(year=int(ret_time[2]), month=int(ret_time[1]), day=int(ret_time[0]), order=-1)

    @staticmethod
    def fix_end(time):
        ret_time = copy(time)
        if time[0] == 'XX':
            ret_time[0] = 31
        if time[1] == 'XX':
            ret_time[1] = 12
        if time[2] == 'XXXX' or time[2] == 'YYYY':
            ret_time[2] = '5000'
        return Time(year=int(ret_time[2]), month=int(ret_time[1]), day=int(ret_time[0]), order=-1)

    def __gt__(self, other):
        if isinstance(other, Time):
            if self.year > other.year:
                return True
            elif self.year == other.year:
                if self.month > other.month:
                    return True
                elif self.month == other.month:
                    if self.day > other.day:
                        return True
                    elif self.day == other.day:
                        if self.order > other.order:
                            return True
            return False
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Time):
            if self.year < other.year:
                return True
            elif self.year == other.year:
                if self.month < other.month:
                    return True
                elif self.month == other.month:
                    if self.day < other.day:
                        return True
                    elif self.day == other.day:
                        if self.order < other.order:
                            return True
            return False
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Time):
            if self.year > other.year:
                return True
            elif self.year == other.year:
                if self.month > other.month:
                    return True
                elif self.month == other.month:
                    if self.day > other.day:
                        return True
                    elif self.day == other.day:
                        if self.order >= other.order:
                            return True
            return False
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Time):
            if self.year < other.year:
                return True
            elif self.year == other.year:
                if self.month < other.month:
                    return True
                elif self.month == other.month:
                    if self.day < other.day:
                        return True
                    elif self.day == other.day:
                        if self.order <= other.order:
                            return True
            return False
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Time):
            return self.year == other.year and self.month == other.month and self.day == other.day
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Time):
            return self.year != other.year or self.month != other.month or self.day != other.day
        return NotImplemented

    def __str__(self):
        return f"{self.day}:{self.month}:{self.year}:{self.order}"


class IComponent(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

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

    def get_encapsulating_interval(self, event):
        ret_interval = None
        for interval in self.timeline:
            if isinstance(interval, Interval):
                if interval.is_encapsulating(event.start, event.end):
                    if ret_interval is None:
                        ret_interval = interval
                    elif ret_interval.start > interval.start or ret_interval.end < interval.end:
                        ret_interval = interval

        return ret_interval

    def get_encapsulating_event(self, event):
        ret_event = None
        for loop_event in self.timeline:
            if isinstance(loop_event, Event):
                if loop_event.is_encapsulating(event.start, event.end):
                    if ret_event is None:
                        ret_event = loop_event
                    elif ret_event.start < loop_event.start or ret_event.end > loop_event.end:
                        ret_event = loop_event

        return ret_event

    def get_all_intervals(self):
        return [interval for interval in self.timeline if isinstance(interval, Interval)]

    def get_all_events(self):
        return [event for event in self.timeline if isinstance(event, Event)]

    def add_event(self, event):
        is_added = False
        interval = self.get_encapsulating_interval(event)
        inner_event = self.get_encapsulating_event(event)
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

    def is_resolved(self):
        intervals = self.get_all_intervals()
        events = self.get_all_events()
        if len(intervals) == 0:
            for event in events:
                if not event.is_resolved():
                    return False

            return True

        return False

    def locate_unresolved_interval(self):
        intervals = self.get_all_intervals()
        if len(intervals) == 0:
            return None

        for interval in intervals:
            move_in = interval.locate_unresolved_interval()
            if move_in is None:
                if not interval.is_resolved():
                    return interval
            else:
                return move_in
        return None

    def __str__(self):
        return f"{self.start}--{self.end}"

    def print_timeline(self, tab=0):
        tabs = '\t' * tab
        tab += 1
        print(f"{tabs}{self}")
        for event_interval in self.timeline:
            if isinstance(event_interval, Interval):
                event_interval.print_timeline(tab)
            else:
                print(f"{tabs}\t{event_interval.text}({event_interval.m_id}) # {event_interval.start}--{event_interval.end}")


class Event(IComponent):
    def __init__(self, text, m_id, start, end):
        super().__init__(start, end)
        self.text = text
        self.m_id = m_id
        self.interval = None

    def set_interval(self, interval):
        self.interval = interval

    def is_resolved(self):
        if self.start.order != -1 and self.end.order != -1:
            return True
        return False

    def __str__(self):
        return f"{self.text}({self.m_id}) # {self.start}--{self.end}"

    def print_timeline(self):
        print(f"{self.text}({self.m_id}) # {self.start}--{self.end}")
