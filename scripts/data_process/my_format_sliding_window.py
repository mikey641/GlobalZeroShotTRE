import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import spacy
from spacy.tokens import Doc


def get_reverse_label(label):
    if label == 'before':
        return 'after'
    elif label == 'after':
        return 'before'
    elif label == 'is_included':
        return 'includes'
    elif label == 'includes':
        return 'is_included'
    else:
        return label


def prettify_xml(elem):
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def process_text(sentences, mentions):
    token_doc_index = 0
    mentions.sort(key=lambda x: x['tokens_ids'][0])
    ment_iter = iter(mentions)
    curr_ment = next(ment_iter)
    for sent_id, sent in enumerate(sentences):
        sent_tokens = list()
        sent_tokens_processed = list()
        start_idx = -1
        end_idx = -1
        for token in sent:
            sent_tokens.append(token.text)
            sent_tokens_processed.append(f'{token.text}///{token.lemma_}///{token.tag_}')
            if curr_ment and curr_ment['tokens_ids'][0] == token_doc_index:
                start_idx = len(sent_tokens) - 1
            if curr_ment and curr_ment['tokens_ids'][-1] == token_doc_index:
                end_idx = len(sent_tokens)
            if start_idx != -1 and end_idx != -1:
                curr_ment['sent_sidx'] = start_idx
                curr_ment['sent_eidx'] = end_idx
                curr_ment['sent_id'] = sent_id
                start_idx = -1
                end_idx = -1

                assert ' '.join(sent_tokens[curr_ment['sent_sidx']:curr_ment['sent_eidx']]) == curr_ment['tokens'], \
                    f"Error: {' '.join(sent_tokens[curr_ment['sent_sidx']:curr_ment['sent_eidx']])} != {curr_ment['tokens']}"
                curr_ment = next(ment_iter, None)

            token_doc_index += 1

    assert curr_ment is None, "Error: Not all mentions were processed"
    return mentions


def process_doc(doc, nlp, only_consecutive=True):
    json_file_data = json.load(open(doc))
    tokens = json_file_data['tokens']
    mentions = json_file_data['allMentions']
    all_pairs = json_file_data['allPairs']
    relv_mentions = []
    for mention in mentions:
        if mention['axisType'] == 'main':
            relv_mentions.append(mention)

    spacy_doc = Doc(nlp.vocab, words=tokens)
    for pipe in filter(None, nlp.pipeline):
        pipe[1](spacy_doc)
    # doc = nlp(text)
    relv_mentions.sort(key=lambda x: x['eventIndex'])
    relv_mentions = process_text(spacy_doc.sents, relv_mentions)

    ment_id_dict = {ment['m_id']: ment for ment in relv_mentions}
    return_pairs = []
    for pair in all_pairs:
        ment_source = ment_id_dict[pair['_firstId']]
        ment_target = ment_id_dict[pair['_secondId']]
        sent_diff = abs(ment_source['sent_id'] - ment_target['sent_id'])
        pair['send_diff'] = sent_diff
        if only_consecutive:
            if sent_diff <= 1:
                return_pairs.append(pair)
        else:
            return_pairs.append(pair)

    ret_new_data = {'tokens': tokens, 'allMentions': relv_mentions, 'allPairs': return_pairs}
    return ret_new_data


def start_process(nlp, input_folder, output_folder):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_folder, file_name)
            ret_data = process_doc(input_file, nlp)
            output_file = os.path.join(output_folder, file_name)
            with open(output_file, 'w') as f:
                json.dump(ret_data, f, indent=4)


if __name__ == '__main__':
    _nlp = spacy.load("en_core_web_trf")
    _input_folder = 'data/NarrativeTime_A2/converted_no_overlap/test'
    _output_folder = 'data/NarrativeTime_A2/converted_no_overlap/test_consecutive_sents'
    start_process(_nlp, _input_folder, _output_folder)
