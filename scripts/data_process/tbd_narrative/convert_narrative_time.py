import json
import os
import re
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from nltk.tokenize import sent_tokenize

from tqdm import tqdm

from scripts.utils.check_trans import evaluate_triplets


class Doc:
    def __init__(self, docid, dct, text):
        self.docid = docid
        self.dct = dct
        self.text = text


class DCT:
    def __init__(self, timex3):
        self.timex3 = timex3


class TIMEX3:
    def __init__(self, tid, type, value, temporalFunction, functionInDocument):
        self.tid = tid
        self.type = type
        self.value = value
        self.temporalFunction = temporalFunction
        self.functionInDocument = functionInDocument


class EVENT:
    def __init__(self, eid, text, tokens_ids, event_class='na'):
        self.eventIndex = -1
        self.axisType = 'na'
        self.rootAxisEventId = -1
        self.corefState = 'unknown'
        self.m_id = eid
        self.tokens = text
        self.tokens_ids = tokens_ids
        self.event_class = event_class


class MAKEINSTANCE:
    def __init__(self, eiid, eventID):
        self.eiid = re.sub("[^0-9]", "", eiid)
        self.eventID = eventID


class TLINK:
    def __init__(self, eventInstanceID, relatedToEvent, relType):
        self.firstEvent = eventInstanceID.replace('ei', '')
        self.secondEvent = relatedToEvent.replace('ei', '')
        self.relType = relType


class CLINK:
    def __init__(self, doc_id, eventInstanceID, relatedToEvent, relType):
        self.doc_id = doc_id
        self.firstEvent = eventInstanceID
        self.secondEvent = relatedToEvent
        self.relType = relType


class MATRES_PAIR:
    def __init__(self, first_event_text, first_event_id, second_event_text, second_event_id, relation):
        self.firstEventText = first_event_text
        self.secondEventText = second_event_text
        self._firstId = first_event_id
        self._secondId = second_event_id
        self._relation = relation


def extract_tlinks(root):
    pairs_set = set()
    event_tlinks = []
    time_tlinks = []
    supported_res = list()
    tot_tlink_elements = len(root.findall('TLINK'))
    for tlink_element in root.findall('TLINK'):
        eventInstanceID = tlink_element.get('eventInstanceID')
        relatedToEventInstance = tlink_element.get('relatedToEventInstance')
        relatedToTime = tlink_element.get('relatedToTime')
        timeId = tlink_element.get('timeID')
        relType = tlink_element.get('relType')
        # Other are cases that its related to time (event->timex, timex->event, timex->timex)
        if f'{eventInstanceID}#{relatedToEventInstance}' in pairs_set or f'{relatedToEventInstance}#{eventInstanceID}' in pairs_set:
            continue

        if eventInstanceID and relatedToEventInstance and relType:
            event_tlinks.append(TLINK(eventInstanceID, relatedToEventInstance, relType))
            supported_res.append(relType)
            pairs_set.add(f'{eventInstanceID}#{relatedToEventInstance}')
            pairs_set.add(f'{relatedToEventInstance}#{eventInstanceID}')
        elif eventInstanceID and relatedToTime and relType:
            time_tlinks.append(TLINK(eventInstanceID, relatedToTime, relType))
        elif timeId and relatedToEventInstance and relType:
            time_tlinks.append(TLINK(timeId, relatedToEventInstance, relType))
        elif timeId and relatedToTime and relType:
            time_tlinks.append(TLINK(timeId, relatedToTime, relType))
        else:
            print(f"ERROR: {eventInstanceID} {relatedToEventInstance} {relatedToTime} {relType}")

    return event_tlinks, time_tlinks, tot_tlink_elements, supported_res


def extract_makeinstances(root):
    makeinstances = []
    for mis in root.findall('MAKEINSTANCE'):
        eiid = mis.get('eiid')
        eventID = mis.get('eventID')
        makeinstances.append(MAKEINSTANCE(eiid, eventID))

    return makeinstances


def get_eiid_from_event_id(event_id, makeinstances):
    for mi in makeinstances:
        if mi.eventID == event_id:
            return mi.eiid

    return None


def parse_timeml(root, doc_id):
    dct_element = root.find('TIMEX3')
    dct = DCT(TIMEX3(dct_element.get('tid'), dct_element.get('type'), dct_element.get('value'),
                     dct_element.get('temporalFunction'), dct_element.get('functionInDocument')))

    doc = Doc(doc_id, dct, None)

    return doc


def sycn_ids(root, events, tokens):
    for mis in root.findall('MAKEINSTANCE'):
        event_id = mis.get('eventID')
        eiid = mis.get('eiid')
        for event in events:
            if event.m_id == event_id:
                event.m_id = eiid.replace('ei', 'e')


def extract_xml_text_as_tokens(root, makeinstances):
    # Parse the XML string
    # root = ET.fromstring(xml_string)

    # Find the <TEXT> element
    events = list()
    text_element = root.find('TEXT')
    if text_element is None:
        return "Error: No <TEXT> element found"

    # Extract all text content, ignoring the tags
    plain_text = ''
    for element in text_element.iter():
        if element.tag == 'EVENT':
            tmp_text = plain_text.replace('\n', ' ')
            tmp_text = tmp_text.strip().split(' ')
            tmp_text = list(filter(None, tmp_text))
            eiid = get_eiid_from_event_id(element.get('eid'), makeinstances)
            new_event = EVENT(eiid, element.text.strip(), list(range(len(tmp_text), len(tmp_text) + len(element.text.strip().split(" ")))), element.get('class'))
            events.append(new_event)
        if element.text:
            tmp_text = element.text.strip().split(' ')
            tmp_text = list(filter(None, tmp_text))
            plain_text += ' ' + " ".join(tmp_text)
        if element.tail:
            tmp_text = element.tail.strip().split(' ')
            tmp_text = list(filter(None, tmp_text))
            plain_text += ' ' + " ".join(tmp_text)

    plain_text = plain_text.replace('\n', ' ')
    plain_text = plain_text.strip().split(' ')
    plain_text = list(filter(None, plain_text))

    # sanity check
    for event in events:
        if event.tokens != ' '.join(plain_text[event.tokens_ids[0]:event.tokens_ids[-1]+1]):
            # if event.tokens == plain_text[event.tokens_ids[0] - 1] and len(event.tokens_ids) == 1:
            #     event.tokens_ids[0] += 1
            # else:
            print(f"ERROR: {event.tokens} != {plain_text[event.tokens_ids[0]:event.tokens_ids[-1]+1]}")

    # sycn_ids(root, events, plain_text)
    return plain_text, events


def read_file(file_path, timeml_parser):
    # Example usage
    tree = ET.parse(file_path)
    root = tree.getroot()
    doc = timeml_parser(root, file_path.name)
    makeinstances = extract_makeinstances(root)
    tokens, events = extract_xml_text_as_tokens(root, makeinstances)
    event_tlinks, time_tlinks, tot_tlink_elements, supported_res = extract_tlinks(root)

    print(f"DOCID: {doc.docid}")
    print(f"num of tokens: {len(tokens)}")
    print(f"num of events: {len(events)}")
    print(f"num of tlinks: {len(event_tlinks)}")

    return doc, tokens, events, event_tlinks, time_tlinks, tot_tlink_elements, supported_res


def convert_relation(rel):
    if rel == 'SIMULTANEOUS':
        ret_rel = 'equal'
    # elif rel == 'VAGUE':
    #     ret_rel = 'uncertain'
    # elif rel == 'IS_INCLUDED':
    #     ret_rel = 'after'
    # elif rel == 'INCLUDES' or rel == "OVERLAP":
    #     ret_rel = 'before'
    else:
        ret_rel = rel.lower()

    return ret_rel


def extract_pair(axis_id, tlink, convertor):
    pair = dict()
    pair['_pairId'] = str(uuid.uuid4())
    pair['_axisId'] = axis_id
    pair['_firstId'] = tlink.firstEvent
    pair['_secondId'] = tlink.secondEvent
    pair['_relation'] = convertor(tlink.relType)

    return pair


def read_matres_file(file_path):
    all_file_data = dict()
    with open(file_path, 'r') as f:
        file_line = f.readlines()
        for line in file_line:
            split = line.split("\t")
            file_name = split[0]

            if file_name not in all_file_data:
                all_file_data[file_name] = list()

            vb1 = split[1].strip()
            vb2 = split[2].strip()
            eiid1 = re.sub("[^0-9]", "", split[3].strip())
            eiid2 = re.sub("[^0-9]", "", split[4].strip())
            rel = split[5].strip()
            all_file_data[file_name].append(MATRES_PAIR(vb1, eiid1, vb2, eiid2, rel))

    return all_file_data


def read_folder(folder_path, timeml_parser):
    total_event_tlinks = 0
    total_time_tlinks = 0
    total_events = 0
    tot_tlink_elements_all = 0
    relation_dist = Counter()
    all_read_files = dict()
    for i, entry in enumerate(os.scandir(folder_path)):
        try:
            doc, tokens, events, event_tlinks, time_tlinks, tot_tlink_elements, supported_res = read_file(entry, timeml_parser)
            relation_dist.update(supported_res)
            total_event_tlinks += len(event_tlinks)
            total_time_tlinks += len(time_tlinks)
            total_events += len(events)
            tot_tlink_elements_all += tot_tlink_elements
            if len(event_tlinks) != ((len(events) * len(events)) - len(events)) / 2:
                print(f"ERROR: {entry.name} has {len(event_tlinks)} missing tlinks for {len(events)} events")

            all_read_files[entry.name] = (doc, tokens, events, event_tlinks)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print(f"Total files: {len(all_read_files)}")
    print(f"Total events: {total_events}")
    print(f"Total event tlinks: {total_event_tlinks}")
    print(f"Total time tlinks: {total_time_tlinks}")
    print(f"Total tlines: {total_event_tlinks + total_time_tlinks}")
    print(f"Total tlines elements: {tot_tlink_elements_all}")
    print(f"Relation distribution: {relation_dist}")
    return all_read_files


def generate(all_tb, convertor):
    my_format = dict()
    for tb_file_name, tb_file_values in tqdm(all_tb.items()):
        tb_file_name = tb_file_name.removesuffix('.tml')

        doc, tokens, events, tlinks = tb_file_values
        event_map = {}
        for event in events:
            event.axisType = 'main'
            event_map[event.m_id] = event

        all_triplets = list()
        relv_events = set()
        event_pair_map = dict()
        all_pairs = list()
        for pair in tlinks:
            found_first = False
            found_second = False

            if pair.firstEvent in event_map:
                found_first = True
                relv_events.add(event_map[pair.firstEvent])
            if pair.secondEvent in event_map:
                found_second = True
                relv_events.add(event_map[pair.secondEvent])

            if found_first and found_second:
                # all_triplets.append((pair.firstEvent, label_set[convertor(pair.relType).upper()], pair.secondEvent))
                if (pair.firstEvent, pair.secondEvent) not in event_pair_map or (pair.secondEvent, pair.firstEvent) in event_pair_map:
                    event_pair_map[(pair.firstEvent, pair.secondEvent)] = True
                    event_pair_map[(pair.secondEvent, pair.firstEvent)] = True
                    all_pairs.append(extract_pair('main', pair, convertor))
            else:
                print(f"ERROR: {pair.firstEvent} or {pair.secondEvent} not found in events")

        my_format[tb_file_name] = {"tokens": tokens, "allMentions": list(relv_events), "allPairs": all_pairs}
        # total_hist, trans_discrepancies, sym_contradictions, error_log = evaluate_triplets(all_triplets, False)
        # if trans_discrepancies > 0:
        #     print(f'ERROR: {tb_file_name}')
        #     print(f"ERROR: {error_log}")

    # TBD -- Filter out symmetric pairs and align the relations
    return my_format


def main():
    narrative_time_path = 'data/NarrativeTime_A2/original'
    all_tb = read_folder(narrative_time_path, parse_timeml)
    output_path = 'data/NarrativeTime_A2/converted'

    conv_my_format = generate(all_tb, convert_relation)
    for doc_id, doc_data in tqdm(conv_my_format.items()):
        with open(f"{output_path}/{doc_id}.json", 'w') as f:
            json.dump(doc_data, f, default=lambda o: o.__dict__, indent=4)

    print(f"Converted {len(all_tb)} files")

if __name__ == '__main__':
    main()
