import json
import os

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from scripts.eval.utils import get_annotations, count_stats_in_file, find_diffs


def create_report(golds, preds, files):
    file_path = f'{output_file}.txt'
    with open(file_path, 'w') as file:
        tmp_pscore_avg = list()
        tmp_rscore_avg = list()
        tmp_fscore_avg = list()
        for file_name in files:
            gold_ments, gold_pairs = golds[file_name]
            pred_ments, pred_pairs = preds[file_name]
            tmp_pscore, tmp_rscore, tmp_fscore, tmp_diff, tmp_unagreed, cause_pscore, cause_rscore, cause_fscore, cause_diff, cause_unagreed = evaluate(gold_ments, gold_pairs, pred_ments, pred_pairs)
            tmp_pscore_avg.append(tmp_pscore)
            tmp_rscore_avg.append(tmp_rscore)
            tmp_fscore_avg.append(tmp_fscore)
            # df = pd.DataFrame(columns=['Mention1', 'Mention2', 'gold', 'predictions'], data=tmp_diff)
            # df_string = df.to_string(index=False)
            # file.write(df_string)
            # file.write('\n')
            file.write(f'{file_name}\n')
            file.write(f'Precision Score: Temporal={tmp_pscore}, Cause={cause_pscore}\n')
            file.write(f'Recall Score: Temporal={tmp_rscore}, Cause={cause_rscore}\n')
            file.write(f'F1 Score: Temporal={tmp_fscore}, Cause={cause_fscore}\n')
            file.write('-------------------------\n')
            # file.write('Most tmp unagreed mentions:\n')
            # file.write('-------------------------\n')
            # for mention, count in dict(sorted(tmp_unagreed.items(), key=lambda item: item[1], reverse=True)).items():
            #     file.write(f'{mention})={count}\n')
            # file.write('#######################################################\n\n')

        file.write(f'Average Precision Score: Temporal={sum(tmp_pscore_avg)/len(tmp_pscore_avg)}\n')
        file.write(f'Average Recall Score: Temporal={sum(tmp_rscore_avg)/len(tmp_rscore_avg)}\n')
        file.write(f'Average F1 Score: Temporal={sum(tmp_fscore_avg)/len(tmp_fscore_avg)}\n')


def evaluate(gold_ments, gold_pairs, pred_ments, pred_pairs):
    average = 'macro'
    tmp_gold, coref_gold, cause_gold = get_annotations(gold_pairs)
    tmp_predictions, coref_predictions, cause_predictions = get_annotations(pred_pairs)

    tmp_gold_unpacked = [item[2] for item in tmp_gold]
    tmp_pred_unpacked = [item[2] for item in tmp_predictions]
    coref_gold_unpacked = [item[2] for item in coref_gold]
    coref_pred_unpacked = [item[2] for item in coref_predictions]
    cause_gold_unpacked = [item[2] for item in cause_gold]
    cause_pred_unpacked = [item[2] for item in cause_predictions]

    tmp_pscore = precision_score(tmp_gold_unpacked, tmp_pred_unpacked, average=average)
    # coref_pscore = precision_score(coref_gold, coref_predictions, average='micro')
    cause_pscore = precision_score(cause_gold_unpacked, cause_pred_unpacked, average=average)
    print(f'Precision Score: Temporal={tmp_pscore}, Cause={cause_pscore}')

    tmp_rscore = recall_score(tmp_gold_unpacked, tmp_pred_unpacked, average=average)
    # coref_rscore = recall_score(coref_gold, coref_predictions, average='micro')
    cause_rscore = recall_score(cause_gold_unpacked, cause_pred_unpacked, average=average)
    print(f'Recall Score: Temporal={tmp_rscore}, Cause={cause_rscore}')

    tmp_fscore = f1_score(tmp_gold_unpacked, tmp_pred_unpacked, average=average)
    # coref_fscore = f1_score(coref_gold, coref_predictions, average='micro')
    cause_fscore = f1_score(cause_gold_unpacked, cause_pred_unpacked, average=average)
    print(f'F1 Score: Temporal={tmp_fscore}, Cause={cause_fscore}')

    tmp_diff, tmp_unagreed = find_diffs(gold_ments, tmp_gold, tmp_predictions)
    cause_diff, cause_unagreed = find_diffs(gold_ments, coref_gold, coref_predictions)
    # coref_diff, coref_unagreed = find_diffs(gold, cause_gold, cause_predictions)

    return tmp_pscore, tmp_rscore, tmp_fscore, tmp_diff, tmp_unagreed, cause_pscore, cause_rscore, cause_fscore, cause_diff, cause_unagreed


def from_pair_to_json(str_pairs):
    splited_pairs = str_pairs.split(",")
    all_metions = set()
    pairs = list()
    for pair in splited_pairs:
        pair_rel = pair.split("=")
        pair_a_b = pair_rel[0].split("/")
        relation = pair_rel[1]

        ment1_text = pair_a_b[0][:pair_a_b[0].find("(")].strip()
        ment2_text = pair_a_b[1][:pair_a_b[1].find("(")].strip()

        ment1_id = pair_a_b[0][pair_a_b[0].find("(") + 1:pair_a_b[0].find(")")]
        ment2_id = pair_a_b[1][pair_a_b[1].find("(") + 1:pair_a_b[1].find(")")]

        ment1 = (ment1_text, ment1_id, "main")
        ment2 = (ment2_text, ment2_id, "main")

        all_metions.add(ment1)
        all_metions.add(ment2)
        pairs.append({"_firstId": ment1_id, "_secondId": ment2_id, "_relation": relation})

    all_ment_as_objs = [{"tokens": ment[0], "m_id": ment[1], "axisType": ment[2]} for ment in all_metions]
    return all_ment_as_objs, pairs


def read_predictions(predictions_before):
    golds = dict()
    preds = dict()
    file_name = None
    for line in predictions_before:
        if line.startswith("----"):
            file_name = None
            continue

        line_split = line.split(":")
        if line_split[0] == "File":
            file_name = line_split[1].strip()
        elif line_split[0] == "Gold":
            gold_pairs = line_split[1].strip()
            golds[file_name] = from_pair_to_json(gold_pairs)
        elif line_split[0] == "Pred":
            pred_pairs = line_split[1].strip()
            preds[file_name] = from_pair_to_json(pred_pairs)
        else:
            print("Not suppose to be here")

    return golds, preds


def main():
    with open(input_file) as f:
        predictions = f.readlines()
        golds, predictions = read_predictions(predictions)

    files = [file for file in golds]
    create_report(golds, predictions, files)


if __name__ == "__main__":
    version = '2'
    model_type = 'gpt4_turbo'
    input_file = f'data/my_data/predictions/output/{model_type}_v{version}.txt'
    output_file = f'data/my_data/predictions/output/{model_type}_report_v{version}'
    main()
