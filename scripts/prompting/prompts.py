def task_description_4rels(examples):
    desc = """
Task Overview:
Analyze the text where events are marked with <eventName(identifier)> and determine temporal relationships (before, after, equal, or vague) based on explicit cues and narrative context.
The context will be followed by the event pairs list requiring evaluation.

Key Guidelines: 
Consider only the starting time of each event as the basis for determining temporal relationships, irrespective of the event's duration or ongoing nature.
Prioritize explicit temporal cues in the text. In cases of ambiguity, rely on reasoning based on the narrative context.
If no explicit cues or clear logical reasoning establish a sequence, label the relationship as vague.
Treat every event marked with a unique identifier (e.g., <eventName(identifier)>) as distinct, regardless of similarities in descriptions. Ensure no identifiers are skipped.

The output should be in two steps:
First: provide a detailed explanation of the story timeline based on your interpretation and the events marked in it.

Then:
For each pair listed below, based on your explanation, provide the temporal relationship between the events in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc


def task_description_6rels(examples):
    desc = """
Task Overview:
Analyze the text where events are marked with <eventName(identifier)> and determine temporal relationships (before, after, equal, includes, is_included, or vague) based on explicit cues and narrative context.
The context will be followed by the event pairs list requiring evaluation.

Key Guidelines: 
Determine relationships based on the starting time, ending time, duration, and ongoing nature of each event.
Label events as includes or is_included when one event occurs within a larger ongoing event.
Prioritize explicit temporal cues in the text. In cases of ambiguity, rely on reasoning based on the narrative context.
If no explicit cues or clear logical reasoning establish a sequence, label the relationship as vague.
Treat every event marked with a unique identifier (e.g., <eventName(identifier)>) as distinct, regardless of similarities in descriptions. Ensure no identifiers are skipped.

The output should be in two steps:
First: provide a detailed explanation of the story timeline based on your interpretation and the events marked in it.

Then:
For each pair listed below, based on your explanation, provide the temporal relationship between the events in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc
