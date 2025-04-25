import csv

from scripts.utils.check_trans import evaluate_triplets


def convert_relation(rel):
    #{"Before": 0, "After": 0, "Equal": 0, "Vague": 0}
    if rel == 'b':
        return 0
    elif rel == 'a':
        return 1
    elif rel == 'e':
        return 2
    elif rel == 'v':
        return 3
    else:
        raise ValueError(f"Unknown relation: {rel}")


def main(timeline_file):
    doc_triplets = dict()
    total_discrepancies = 0
    doc_with_discrepancies = 0
    with open(timeline_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row[0] == '﻿Document ID':
                continue

            if row[0] not in doc_triplets:
                doc_triplets[row[0]] = list()
            doc_triplets[row[0]].append((row[3], convert_relation(row[5]), row[4]))

    for doc_id, triplets in doc_triplets.items():
        print(f"Document ID: {doc_id}")
        total_hist, trans_discrepancies, sym_contradictions, error_log = evaluate_triplets(triplets, None, False)
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
    _timeline_file = "data/TimeLine/Annotated_Relations.csv"
    main(_timeline_file)
