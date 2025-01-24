import json
from tqdm import tqdm

from convert_narrative_time import DCT, TIMEX3, Doc, generate, read_folder


def parse_timeml(root, doc_id):
    dtc = root.find('DCT')
    dct_element = dtc.find('TIMEX3')
    dct = DCT(TIMEX3(dct_element.get('tid'), dct_element.get('type'), dct_element.get('value'),
                     dct_element.get('temporalFunction'), dct_element.get('functionInDocument')))

    doc = Doc(doc_id, dct, None)

    return doc


def convert_relation(rel):
    if rel == 'SIMULTANEOUS':
        ret_rel = 'equal'
    elif rel == 'NONE':
        ret_rel = 'uncertain'
    else:
        ret_rel = rel.lower()

    return ret_rel


def main():
    timebank_path = 'data/TimeBank-Dense/all_in_one_folder'
    all_tb = read_folder(timebank_path, parse_timeml)
    output_path = 'data/TimeBank-Dense/all_converted_tmp'

    conv_my_format = generate(all_tb, convert_relation)
    for doc_id, doc_data in tqdm(conv_my_format.items()):
        with open(f"{output_path}/{doc_id}.json", 'w') as f:
            json.dump(doc_data, f, default=lambda o: o.__dict__, indent=4)

    print(f"Converted {len(all_tb)} files")

if __name__ == '__main__':
    main()
