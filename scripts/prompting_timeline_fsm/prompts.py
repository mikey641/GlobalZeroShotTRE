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
    Provide the timeline order of the following events: """ + events + """, with a numerical indicator. If two events started at the same time, assign them the same order indicator. 
    Output in JSON format, where the key is the event (with identifier) and the value is the indicator.
     
    The output should be in two steps:
    First: provide a detailed explanation of the timeline based on your interpretation and the events.

    Then, provide the JSON output.
    """

    return desc
