def extract_times():
    desc = """
    For each marked event (i.e., <event(identifier)>) in the text below, state the date/time it took place (as precisely as possible) in an easy-to-post-process JSON format 
    (i.e., the keys are the event(identifier), and the values should be in ISO format DD:MM:YYYY--DD:MM:YYYY, indicating the start--end of the event. 
    Use XX for unknown days or months; however, the year should be specified).
    """

    return desc


def extract_timeline(events, pairs):
    # events = called(18), smacked(11), asked(4)
    # pairs = called(18) - - smacked(11)
    #         called(18) - - asked(4)
    #         smacked(11) - - asked(4)
    desc = """
    Provide the relations (before, after, equal, vague) between the events """ + events + """, according to the list of pairs below. 
    The output should be in two steps:
    First: provide a detailed explanation of the timeline based on your interpretation and the events.

    Then:
    For each pair listed below, based on your explanation, provide the temporal relationship between the events in the following DOT format:
    strict graph {
    "Event1(id)" -- "Event2(id)" [rel=LABEL];
    "Event1(id)" -- "Event3(id)" [rel=LABEL];
    ...
    }

    Pairs require classification:
    """ + pairs

    return desc
