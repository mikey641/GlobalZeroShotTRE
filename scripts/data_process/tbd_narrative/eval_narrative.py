import csv
import json
import os

from nltk.internals import Counter

from scripts.utils.check_trans import evaluate_triplets


def convert_relation(rel):
    #{"Before": 0, "After": 0, "Equal": 0, "Vague": 0}
    if rel == 'before':
        return 0
    elif rel == 'after':
        return 1
    elif rel == 'equal':
        return 2
    elif rel == 'vague':
        return 3
    elif rel == 'includes':
        return 4
    elif rel == 'is_included':
        return 5
    elif rel == 'overlap':
        return 6
    else:
        raise ValueError(f"Unknown relation: {rel}")


def main(narrative_folder):
    doc_triplets = dict()
    total_discrepancies = 0
    doc_with_discrepancies = 0
    # read all files in folder
    for nar_file in os.listdir(narrative_folder):
        with open(f'{narrative_folder}/{nar_file}', mode='r') as file:
            data = json.load(file)
            all_pairs = data['allPairs']
            for pair in all_pairs:
                if nar_file not in doc_triplets:
                    doc_triplets[nar_file] = list()
                doc_triplets[nar_file].append(
                    (pair['_firstId'], convert_relation(pair['_relation']), pair['_secondId']))

    for doc_id, triplets in doc_triplets.items():
        print(f"Document ID: {doc_id}")
        total_hist, trans_discrepancies, sym_contradictions, error_log = evaluate_triplets(triplets, False)
        total_discrepancies += trans_discrepancies
        print(f"Total History: {total_hist}")
        print(f"Total Discrepancies: {trans_discrepancies}")
        if trans_discrepancies > 0:
            doc_with_discrepancies += 1
        # print(f"Total Contradictions: {sym_contradictions}")
        print(f"Error Log: {error_log}")

    print(f"Total Discrepancies All Docs: {total_discrepancies}")
    print(f"Total Docs with Discrepancies: {doc_with_discrepancies}")
    return doc_triplets


if __name__ == '__main__':
    _narrative_file = "data/NarrativeTime/converted"
    main(_narrative_file)
