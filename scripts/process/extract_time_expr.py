import json
import os
import re

from openai import OpenAI

# Instructions: Highlight any temporal expressions, including but not limited to:
# Days, months, and years: e.g., 5/5/22, 05-05-22
# Specific dates: e.g., July 4th, 1990
# Hours and minutes: e.g., 5pm, 17:00
# Time periods during the day: e.g., morning, afternoon, evening, night
# Recurring time frames: e.g., every week, once a week, biweekly, daily
# Durations: e.g., from June to May, during March, for two hours, all day
# References to time: e.g., now, today, tomorrow, next week, last year
# Relative times: e.g., in two days, three weeks ago, later, soon
# Holidays and special days: e.g., Christmas, New Year's Day, Halloween
# Seasons: e.g., summer, winter, spring, fall
# Decades and centuries: e.g., the 1990s, the 21st century
# Note: Do not highlight numbers that are not related to time, such as amounts, currency, age, or quantities.
#


def task_description():
    desc = """
    Please highlight in bold all temporal entities in the following text. This includes, but is not limited to:
    1) Time Expressions: Specific points in time, such as "Saturday," "November 2010," "2015," etc.
    2) Temporal Prepositions: Prepositions that indicate time, such as "on," "between," "in," "before," "after," etc.
    3) Temporal Conjunctions: Conjunctions that relate to time, such as "when," "while," "since," "until," "as soon as," etc.
    4) Temporal Adverbs: Adverbs describing when something happens, such as "now," "soon," "later," "recently," "yesterday," "today," "tomorrow," etc.
    5) Temporal Clauses: Subordinate clauses specifying the time of an action, e.g., "when I arrived," "after the meeting ended," "before she left."
    6) Temporal Nouns: Nouns referring to specific periods or points in time, such as "morning," "evening," "day," "week," "month," "year," "century," "era."
    7) Temporal Phrases: Phrases specifying time, such as "at the moment," "in the past," "for a while," "during the night," "by next week."
    8) Time Periods and Durations: Expressions describing lengths of time, such as "a decade," "a few minutes," "several hours," "three years."
    9) Calendar References: Specific dates or named periods, such as "January 1st," "Christmas," "the 21st century," "the Renaissance."
    10) Frequency Adverbs: Adverbs describing how often something happens, such as "always," "never," "often," "rarely," "sometimes," "usually."
    Here's the text to analyze:
    """
    return desc


def run_gpt4o_turbo(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def extract_time_expr(data):
    tokens = data['main_doc']['tokens']
    text = ' '.join(tokens)
    task_desc = task_description() + text
    response = run_gpt4o_turbo(task_desc)
    # response = text_file

    time_exprs = set()
    # find all text that is highlighted between ** and **, extract the indexes and the text. once extracted move to the next ** pair.
    for match in re.finditer(r'\*\*(.*?)\*\*', response):
        time_exprs.add(match.group(1))

    data['main_doc']['time_exprs'] = list()
    for time_expr in time_exprs:
        time_expr_text = time_expr.split(" ")
        len_tokens = len(tokens)
        len_inner_expr = len(time_expr_text)
        result = []

        for i in range(len_tokens - len_inner_expr + 1):
            if tokens[i:i + len_inner_expr] == time_expr_text:
                result.append(list(range(i, i + len_inner_expr)))

        data['main_doc']['time_exprs'].append({
            'text': time_expr,
            'indices': result
        })

    return data


def main():
    # iterate over all files in data/time/without_time/
    # for file in os.listdir('data/time/without_time/'):
    with open(f'data/tmp/input_example.json', 'r') as f:
        data = json.load(f)

    new_data = extract_time_expr(data)

    with open(f'data/tmp/input_example_withtime.json', 'w') as f:
        json.dump(new_data, f, indent=4)


if __name__ == '__main__':
    main()
