import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import spacy
from spacy.tokens import Doc


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
                curr_ment['sent_toks'] = sent_tokens
                curr_ment['sent_toks_proc'] = sent_tokens_processed
                curr_ment['sent_id'] = sent_id
                start_idx = -1
                end_idx = -1

                assert ' '.join(sent_tokens[curr_ment['sent_sidx']:curr_ment['sent_eidx']]) == curr_ment['tokens'], \
                    f"Error: {' '.join(sent_tokens[curr_ment['sent_sidx']:curr_ment['sent_eidx']])} != {curr_ment['tokens']}"
                curr_ment = next(ment_iter, None)

            token_doc_index += 1

    assert curr_ment is None, "Error: Not all mentions were processed"
    return mentions


def generate_xml_sents(mentions, pairs, data, doc_id):
    ment_dict = {ment['m_id']: ment for ment in mentions}
    for p in pairs:
        ment1 = ment_dict[p['_firstId']]
        ment2 = ment_dict[p['_secondId']]
        DOCID = doc_id
        SOURCE = p['_firstId']
        TARGET = p['_secondId']
        SOURCE_SENTID = str(ment1['sent_id'])
        TARGET_SENTID = str(ment2['sent_id'])
        LABEL = p['_relation'].upper()
        LABEL = 'VAGUE' if LABEL == 'UNCERTAIN' else LABEL
        SENTDIFF = str(abs(ment1['sent_id'] - ment2['sent_id']))

        ment_1_start = ment1['sent_sidx']
        ment_1_end = ment1['sent_eidx']
        ment_2_start = ment2['sent_sidx']
        ment_2_end = ment2['sent_eidx']
        final_tokens = list()
        if SOURCE_SENTID == TARGET_SENTID:
            for idx, tok in enumerate(ment1['sent_toks_proc']):
                if idx < ment_1_start:
                    final_tokens.append(f'{tok}///B')
                elif ment_1_start <= idx < ment_1_end:
                    final_tokens.append(f'{tok}///E1')
                elif ment_1_end <= idx < ment_2_start:
                    final_tokens.append(f'{tok}///M')
                elif ment_2_start <= idx < ment_2_end:
                    final_tokens.append(f'{tok}///E2')
                else:
                    final_tokens.append(f'{tok}///A')
        else:
            for idx, tok in enumerate(ment1['sent_toks_proc']):
                if idx < ment_1_start:
                    final_tokens.append(f'{tok}///B')
                elif ment_1_start <= idx < ment_1_end:
                    final_tokens.append(f'{tok}///E1')
                else:
                    final_tokens.append(f'{tok}///M')

            for idx, tok in enumerate(ment2['sent_toks_proc']):
                if idx < ment_2_start:
                    final_tokens.append(f'{tok}///M')
                elif ment_2_start <= idx < ment_2_end:
                    final_tokens.append(f'{tok}///E2')
                else:
                    final_tokens.append(f'{tok}///A')

        text = ' '.join(final_tokens)

        sentence_element = ET.SubElement(data, 'SENTENCE',
                                         DOCID=DOCID,
                                         SOURCE=SOURCE,
                                         TARGET=TARGET,
                                         SOURCE_SENTID=SOURCE_SENTID,
                                         TARGET_SENTID=TARGET_SENTID,
                                         LABEL=LABEL,
                                         SENTDIFF=SENTDIFF)
        sentence_element.text = text


def process_doc(doc, nlp, xml_data, doc_id):
    json_file_data = json.load(open(doc))
    tokens = json_file_data['tokens']
    mentions = json_file_data['allMentions']
    all_pairs = json_file_data['allPairs']
    relv_mentions = []
    for mention in mentions:
        if mention['axisType'] == 'main':
            # tokens[mention['tokens_ids'][0]] = "{%s" % tokens[mention['tokens_ids'][0]]
            # tokens[mention['tokens_ids'][-1]] = "%s}" % tokens[mention['tokens_ids'][-1]]
            relv_mentions.append(mention)

    # text = ' '.join(tokens)
    spacy_doc = Doc(nlp.vocab, words=tokens)
    for pipe in filter(None, nlp.pipeline):
        pipe[1](spacy_doc)
    # doc = nlp(text)
    relv_mentions.sort(key=lambda x: x['eventIndex'])
    relv_mentions = process_text(spacy_doc.sents, relv_mentions)
    generate_xml_sents(relv_mentions, all_pairs, xml_data, doc_id)


def start_process(nlp, input_folder, output_file):
    data = ET.Element('DATA')
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_folder, file_name)
            process_doc(input_file, nlp, data, file_name)

    pretty_xml = prettify_xml(data)
    with open(output_file, "w") as f:
        f.write(pretty_xml)


if __name__ == '__main__':
    _nlp = spacy.load("en_core_web_trf")
    _input_folder = 'data/TimeBank-Dense/test_converted'
    _output_file = 'data/bayesian_format/testset_tbd.xml'
    start_process(_nlp, _input_folder, _output_file)
