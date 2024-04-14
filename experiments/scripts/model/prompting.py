import json
import os

import google.generativeai as genai
from openai import OpenAI


def run_gpt4(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def run_gpt3_5(_prompt):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": _prompt
            },
        ]
    )

    response_content = response.choices[0].message.content
    return response_content


def run_gemini_pro(_prompt):
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    # for m in genai.list_models():
    #     if 'generateContent' in m.supported_generation_methods:
    #         print(m.name)
    response = model.generate_content(_prompt)
    return response.text


def task_description_v1():
    # this example is generated from the 139d3_temporal_Alon.json file
    desc = """
    Task Overview:
    You will be provided with a text containing events marked by '<' and '>' symbols. Your primary task is to construct 
    a temporal graph that maps out all the temporal relationships among these events. 
    To do so, first identify all the events in the text and order them as a list by their appearance order in the text. 
    Then determine the temporal relationships between each pair of events. 
    The temporal relationships will be based on the starting times of the events.
    
    Event Relationships:
    For each pair of events, you will assign one of the following temporal relationships based on the starting times of the events:
    
    Before: Event A started before Event B. For example, given the text "A traveler is <kidnapped>, and the police officers <said> the kidnapper is demanding money," 'kidnapped' started before 'said', therefore in the matrix cell representing 'kidnapped' and 'said' you should put the relation 'before'.
    After: Event B started before Event A. Using the same example, but with the cell referring to 'said' and 'kidnapped', you should put the relation 'after', as 'said' happened after 'kidnapped'.
    Equal: Event A and B started simultaneously. For example, given the text "They <filed> objections to the court, <claiming> that the suspects were treated unfairly," both 'filed' and 'claiming' happened at the same time, therefore the relation between them should be 'equal'.
    Uncertain: The order of events cannot be determined from the context. For example, given the text "I <ate> a burger and <drank> a bottle of water for lunch today," we only know that 'ate' and 'drank' happened during lunchtime. We cannot ascertain from the text whether 'ate' is earlier or later than 'drank', so the relation between them should be 'uncertain'.
    
    Output Format:
    You will produce a list of mentions and a matrix representing the relationships between all marked events. 
    The matrix is to be read from left to right and top to bottom, with each index corresponding to an event's chronological appearance in the text. 
    
    Example:
    Consider this sample text with marked events:
    – " We 've just been Banksy'ed . " So declared Sotheby 's of London on Friday after a bizarre < stunt > apparently pulled off by the elusive artist himself — one of his prints < shredded > itself just after being < sold > at auction for $ 1.4 million . Banksy has since < revealed > how the < shredding > of his 2006 " Girl With Balloon " took place , but some think the artist may have inadvertently revealed more than that : perhaps an image of himself , or at the very least a close assistant . Details and developments : The ' how ' : Banksy posted a video online showing that he < built > a remote - controlled shredder into the print 's frame " a few years ago " so he could destroy the work if it ever went up for auction . The video also captures the initial moments of that happening at Sotheby 's , and this video via USA Today shows more reaction from the auction . The ID buzz : The video posted by Banksy appears to be taken from the vantage point of a man who was pictured at Sotheby 's filming the scene . You can see an image of him at Lad Bible . He 's a middle - aged man with curly hair , and countless online speculators point out that he resembles a street artist named Robin Gunningham , one of the leading suspects in the who - is - Banksy question . " All of this is of course speculation but when it comes to Banksy , let 's face it , everything is , " observes a post about the auction 's " mystery man " at Sky News . ID buzz , II : Caroline Lang , chief of Sotheby 's Switzerland , posted an image of another man who appeared to be activating a remote - control device , and she identified him as Banksy , reports the New York Times . Alas , the account is private and the photo unavailable . Inside job ? Sotheby 's swears it was n't in on the stunt , but some are skeptical . One dealer tells the Times that he pointed out to staff before the auction that the print 's frame was weirdly large , but they had no explanation . " If the upper management knew , I ca n't speculate . " Plus , the print was < placed > in a relatively hard - to - access viewing spot before the < auction > , then was < sold > dead last in the 67 - item sale , which was " odd , " he says . Good question : If Banksy did indeed embed the shredder a " few years ago , " would n't the battery have needed replacement since then ? So wonders Scott Reyburn at the Times . The irony : Banksy may have been making a statement about what he views as absurd prices for his work , but the stunt likely increased the value of the print , writes Leonid Bershidsky at Bloomberg . Of course , Banksy surely knew that would happen , this being only his " latest contribution to the empirical study of the value of art . " That 's where his true genius lies , writes Bershidsky . Another take : Sebastian Smee at the Washington Post also digs into Banksy 's motivations and theme of destruction in avant - garde art . So what 's the main problem in all this ? " Is it a system that values art in monetary terms in order for it to be exchanged on the market ? Or is it a system in thrall to the currency of publicity and self - promotion ? If it ’s the latter , Banksy is deeply implicated . "
    
    The provided sample text includes nine marked events. Your output will be a 9x9 matrix where each row and column index corresponds to a mention in the order they appear, starting from index 0. 
    Only fill the upper triangle of the matrix with the designated relationships, leaving the lower triangle blank (marked with zeros).
    
    The output should be:
    stunt, shredded, sold, revealed, shredding, built, placed, auction, sold
    0, equal, after, before, equal, after, after, after, after
    0, 0, after, before, equal, after, after, after, after
    0, 0, 0, before, before, after, after, after, equal
    0, 0, 0, 0, after, after, after, after, after
    0, 0, 0, 0, 0, after, after, after, after
    0, 0, 0, 0, 0, 0, before, before, before
    0, 0, 0, 0, 0, 0, 0, before, before
    0, 0, 0, 0, 0, 0, 0, 0, before
    0, 0, 0, 0, 0, 0, 0, 0, 0
    
    Notes:
    Consider only the starting times of the events when determining their temporal relationship.
    Ensure to review the entire context of the text to best understand the sequence of events.
    Accuracy is crucial, as the output will contribute to building a comprehensive temporal graph of the event sequence.
    Output only the mentions list and the upper triangle of the matrix, without any additional information.
    
    You input text with events to process:
    
    """
    return desc


def filter_non_events(events):
    return [e for e in events if e['axisType'] == 'main']


def mark_events_in_text(tokens, all_mentions):
    for mention in all_mentions:
        tok_first_id = mention['tokens_ids'][0]
        tok_last_id = mention['tokens_ids'][-1]
        tokens[tok_first_id] = f'< {tokens[tok_first_id]}'
        tokens[tok_last_id] = f'{tokens[tok_last_id]} >'
    return " ".join(tokens)


def get_example_matrix(pairs, all_ment_ids):
    example_matrix = [[0 for _ in range(len(all_ment_ids))] for _ in range(len(all_ment_ids))]
    for pair in pairs:
        first_id = pair['_firstId']
        second_id = pair['_secondId']
        relation = pair['_relation']
        if '/' in relation:
            split_rel = relation.split('/')
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = split_rel[0]
        else:
            example_matrix[all_ment_ids.index(first_id)][all_ment_ids.index(second_id)] = relation

    # pretty print matrix
    print("Expected matrix:")
    for row in example_matrix:
        print(row)

    return example_matrix


def main():
    # load json file
    with open('data/my_data/michael_netta/135d4_temp_netta_v2.json') as file:
        data = json.load(file)

    tokens = data['tokens']
    all_mentions = filter_non_events(data['allMentions'])
    all_mentions.sort(key=lambda x: x['tokens_ids'][0])
    all_mentions_text = [m['tokens'] for m in all_mentions]
    all_ment_ids = [m['m_id'] for m in all_mentions]
    all_pairs = data['allPairs']
    text = mark_events_in_text(tokens, all_mentions)

    num_of_mentions_txt = f'The following text contains-{len(all_mentions)} mentions \n'
    print(num_of_mentions_txt)
    print(f'The mentions are-{all_mentions_text}')
    print(f'The input text is-{text}')

    get_example_matrix(all_pairs, all_ment_ids)
    task_desc = task_description_v1() + num_of_mentions_txt + text
    response = llm_to_use(task_desc)

    print()
    print("Predicted matrix:")
    print(response)


if __name__ == "__main__":
    # llm_to_use = run_gemini_pro
    llm_to_use = run_gpt4
    main()
