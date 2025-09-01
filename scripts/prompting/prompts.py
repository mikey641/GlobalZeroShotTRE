def task_description_4res_only_global(examples):
    desc = """
Given the document below where each event is marked with <eventName(identifier)>, for each pair of events listed below, determine the temporal relationships (before, after, equal, vague) between them. 
Answer in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc


def task_description_4res_only_timeline(examples):
    desc = """
Given the document below where each event is marked with <eventName(identifier)>, for each pair of events listed below, determine the temporal relationships (before, after, equal, vague) between them. 

The output should be in two steps:
First: provide an explanation of the story timeline based on the events marked in it.

Then:
Based on your explanation, provide the temporal relationship between the events in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc


def task_description_6res_only_global(examples):
    desc = """
Given the document below where each event is marked with <eventName(identifier)>, for each pair of events listed below, determine the temporal relationships (before, after, equal, includes, is_included, vague) between them. 
Answer in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc


def task_description_6res_only_timeline(examples):
    desc = """
Given the document below where each event is marked with <eventName(identifier)>, for each pair of events listed below, determine the temporal relationships (before, after, equal, includes, is_included, vague) between them. 

The output should be in two steps:
First: provide a detailed explanation of the story timeline based on the events marked in it.

Then:
Based on your explanation, provide the temporal relationship between the events in the following DOT format:

strict graph {
"Event1(id)" -- "Event2(id)" [rel=LABEL];
"Event1(id)" -- "Event3(id)" [rel=LABEL];
...
}
"""
    return desc
