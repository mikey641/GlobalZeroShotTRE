def task_description_v1(examples):
    # this example is generated from the 139d3_temporal_Alon.json file
    desc = f"""
Task Overview:
You will be given a text with event mentions highlighted within it (identified by '<' and '>' symbols),
Each event mention in the text is coupled with its mention ID immediately after (in parentheses).  
Your task is to determine the temporal relationships between all given pairs based on the provided text and output a temporal relation graph.
Output only the temporal relationships between the pairs in DOT format, without any additional information.

Instructions:
Consider only the starting time of events to determine the temporal relationships between them.

Event Relationships:
For each pair of events, you will assign one of the following temporal relationships based on the starting times of the events:

Before: Event A started before Event B. For example, given the text "A traveler is <kidnapped>, and the police officers <said> the kidnapper is demanding money,". Event 'kidnapped' started before 'said', therefore the pair 'kidnapped'/'said' should equal 'before'.
After: Event A started before Event B. Using the same example, for the pair 'said'/'kidnapped', you should now put the relation 'after', as 'said' happened after 'kidnapped'.
Equal: Event A and B started at the same time. For example, given the text "They <filed> objections to the court, <claiming> that the suspects were treated unfairly". both 'filed' and 'claiming' are refering to the same thing and therefore happened at the same time, the relation between 'filed'/'claiming' or 'claiming'/'filed' should be 'equal'.
Vague: The order of events cannot be determined from the context. For example, given the text "I <ate> a burger and <drank> a bottle of water for lunch today,". We cannot ascertain from the text whether 'ate' is earlier or later than 'drank', so the relation between 'ate'/'drank' should be 'vague'.

To accurately assign the correct temporal relationships between events, look for explicit temporal indicators in the text. 
These indicators may include specific dates, times, and keywords such as 'before', 'after', 'following, 'at the same time', etc. 

Example(s):
{examples}

Following is the input text with events for you to process:
    """
    return desc


def task_description_v2(examples):
    desc = f"""
Task Overview:
You will analyze a text where events are marked with '<' and '>' symbols, and each event is identified with an ID shown in parentheses immediately after the event. 
Your task is to determine the temporal relationships between these pairs based on the starting times of the event mentions.
Output only the temporal relationships between the pairs in DOT format, without any additional information.

Instructions for Determining Temporal Relationships:
Before: If Event A starts before Event B. Example: "A traveler is <kidnapped(01)>, and the police officers <said(02)> The kidnapper is demanding money". 
Since 'kidnapped' must have occurred before 'said', the relationship should be kidnapped 'before' said.
After: If Event A starts after Event B. Using the previous example, the relationship for 'said(02) and kidnapped(01)' is 'after'.
Equal: If Event A and B start simultaneously, for example: "They <filed(03)> objections, <claiming(04)> unfair treatment". Since both events refer to the same thing, they occured at the same time, thus, the relationship should be 'equal'.
Vague: If the start times of Event A and B cannot be determined, for example: "I <ate(05)> and <drank(06)> at lunch". The sequence cannot be determined if I first started to eat or first started to drink as both are logically possible, the relationship should be 'vague'.

Look for Temporal Indicators:
Use explicit temporal indicators such as dates, times, and keywords like 'before', 'after', and 'at the same time' in the provided context to help determine relationships when logical deduction fails. 
If no indicators are present, and logical deduction fails, mark the relationship as 'vague'.

Example(s):
{examples}

Following is the input text with events for you to process and predict:
    """
    return desc


def task_description_v3(examples):
    desc = f"""
Task Overview:
You will analyze a text where events are marked with '<' and '>' symbols, and each event is identified with an ID shown in parentheses immediately after the event. 
Your task is to determine the temporal relationships between these pairs based on the starting times (only) of the events.

Output only the temporal relationships between the pairs in DOT format, without any additional information.

Instructions for Determining Temporal Relationships:
Before: If Event A starts before Event B. Example: "A traveler is <kidnapped(01)>, and the police officers <said(02)> The kidnapper is demanding money". 
Since 'kidnapped' must have occurred before 'said', the relationship should be kidnapped 'before' said.
After: If Event A starts after Event B. Using the previous example, the relationship for 'said(02) and kidnapped(01)' is 'after'.
Equal: If Event A and B start simultaneously, for example: "They <filed(03)> objections, <claiming(04)> unfair treatment". Since both events refer to the same thing, they occured at the same time, thus, the relationship should be 'equal'.
Vague: If the start times of Event A and B cannot be determined, for example: "I <ate(05)> and <drank(06)> at lunch". The sequence cannot be determined if I first started to eat or first started to drink as both are logically possible, the relationship should be 'vague'.

Maintaining Transitive Consistency
Before transitivity:
If A is before B, and B is before C, then A is before C.
If A is before B, and B equals C, then A is before C.

After transitivity:
If A is after B, and B is after C, then A is after C.
If A is after B, and B equals C, then A is after C.

Equal relationships:
If A equals B, and B equals C, then A equals C.
If A equals B, and B is before C, then A is before C.
If A equals B, and B is after C, then A is after C.

Vague handling:
If A is vague B, and B equals C, then A is vague C.
If A equals B, and B is vague C, then A is vague C.

Look for Temporal Indicators:
Use explicit temporal indicators such as dates, times, and keywords like 'before', 'after', and 'at the same time' in the provided context to help determine relationships when logical deduction fails. 
If no indicators are present, and logical deduction fails, mark the relationship as 'vague'.

Pay close attention to these instructions and examples to accurately determine the temporal relationships between events.

{examples}

Following is the input text with events for you to process and predict:
    """
    return desc


def task_description_v5_with_example(examples):
    desc = f"""
Task Overview:
Analyze the text where events are marked with <eventName(identifier)> and determine temporal relationships (before, after, equal, or vague) based on explicit cues and narrative context.

Key Guidelines: 
Consider only the starting time of each event as the basis for determining temporal relationships, irrespective of the event's duration or ongoing nature.
Prioritize explicit temporal cues in the text. In cases of ambiguity, rely on reasoning based on the narrative context.
If no explicit cues or clear logical reasoning establish a sequence, label the relationship as vague.
Treat every event marked with a unique identifier (e.g., <eventName(identifier)>) as distinct, regardless of similarities in descriptions. Ensure no identifiers are skipped.

The output should be in two steps:
First: provide a detailed explanation of the story timeline based on your interpretation and the events marked in it.

Then:
For each pair, based on your explanation, provide the temporal relationship between the events in DOT format.
{examples}

Following is the input text with events for you to process and predict:
    """
    return desc


def task_description_v5(examples):
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


def task_description_tbd(examples):
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
