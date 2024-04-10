import json
import pathlib
import textwrap

import google.generativeai as genai

GOOGLE_API_KEY = "<YOUR_KEY_HERE>"

genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

model = genai.GenerativeModel('gemini-pro')

# load json file
with open('data/Multi_News/2d7.json') as file:
    data = json.load(file)


event_def = """We define an event as any occurrence, action or process 
which deserves a place upon a timeline, and could have any syntactic realization – as verbs, nominalizations,
nouns. You should focus on the semantic questions of what is actually happening.

Events constitute a single word (in case of a phrase it should be the head of the phrase), 
the only exception are proper-names which are used to describe events names.

Examples (events are highlighted with *asterisks*): 
- Your dog seems to have *eaten* the cupcakes
- The *storm* caused a lot of damage
- The *election* was very close
- *Hurricane Katrina* was a *disaster*
- The *storm* was a *disaster*

We should not consider the following as reporting events as events:
Non-events (non-events are highlighted with *asterisks*):
- CNN *reported* that the ...
- He *said* that the ...
- was *quoted* as *saying*

"""

json_format = """
{
    "m_id": 0,
    "tokens_ids": [
        80
    ],
    "tokens": "instituted"
}
"""

tokens = data["main_doc"]["tokens"]
prompt = (f'Your task is as follow: given this list of tokens representing the text of a news article {str(tokens)},'
          f' and the following definition of events: {event_def}. '
          f'Your task is to identify the events in the text, and extract them as a list '
          f'of json objects, where each json object is in the following format {json_format} '
          f'(m_id is a unique event mention id, tokens_ids is the list of token ids of the event in the token list, '
          f'and tokens is the text of the event.')

response = model.generate_content(prompt)
print(response.text)
