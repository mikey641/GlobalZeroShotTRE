def extract_times():
    desc = """
For each marked event (i.e., <event(identifier)>) in the text below, state the date/time it took place (as precisely as possible) in an easy-to-post-process JSON format 
(i.e., the keys are the 'event(identifier)', and the values should be in ISO format DD:MM:YYYY--DD:MM:YYYY, indicating the start--end of the event. 
Use XX for unknown days or months; however, the year should be specified).
"""

    return desc


def extract_times_missing_events(events):
    desc = """
The following events where missed: """ + events + """.
As instructed previously, for each of these events, state the date/time it took place in JSON format.
"""
    return desc


def extract_timeline_order():
    desc = """
Now provide the timeline order between the events, the order should be indicated with a numerical indicator. 
If two events started at the same time, assign them the same order indicator. If the event order is uncertain, assign 'X' as the order indicator. 
Output in JSON format, where the key is the event (with identifier) and the value is the indicator.
"""
    return desc


def extract_timeline(events):
    # events = called(18), smacked(11), asked(4)
    # pairs = called(18) - - smacked(11)
    #         called(18) - - asked(4)
    #         smacked(11) - - asked(4)
    desc = """
Provide the timeline order of the following events: """ + events + """, with a numerical indicator. 
If two events started at the same time, assign them the same order indicator. If the event order is uncertain, assign 'X' as the order indicator. 
Output in JSON format, where the key is the event (with identifier) and the value is the indicator.
 
The output should be in two steps:
First: provide a detailed explanation of the timeline based on your interpretation and the events.

Then, provide the JSON output.
"""

    return desc


def extract_missing_events_order(events):
    desc = """
The timeline order of the following events was missed: """ + events + """.  
As previously instructed, for each of these events, provide its order (considering the sequence of events you generated earlier) in JSON format.
"""
    return desc


def extract_relations(events, pairs):
    desc = """
Provide the temporal relations (before, after, equal, vague) between the following events: """ + events + """. 
Assign vague if the relation is uncertain. 

The output should be in two steps:
First: provide a detailed explanation of the timeline based on your interpretation and the targeted events.

Then:
For each pair listed below, based on your explanation, provide the temporal relationship between the events in the following Json format:
[
    {
        "source": event1(identifier),
        "target": event2(identifier),
        "relation": temporal_relation
    },
    ...
]

Pairs require classification:
""" + pairs

    return desc
