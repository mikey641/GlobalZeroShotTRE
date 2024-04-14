import json

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from scripts.eval.utils import get_annotations, count_stats_in_file, find_diffs


def create_report(gold_ments, gold_pairs, pred_ments, pred_pairs):
    tmp_pscore, tmp_rscore, tmp_fscore, tmp_diff, tmp_unagreed, cause_pscore, cause_rscore, cause_fscore, cause_diff, cause_unagreed = evaluate(gold_ments, gold_pairs, pred_ments, pred_pairs)
    df = pd.DataFrame(columns=['Mention1', 'Mention2', 'gold', 'predictions'], data=tmp_diff)
    df_string = df.to_string(index=False)
    file_path = f'{output_file}.txt'
    with open(file_path, 'w') as file:
        file.write(df_string)
        file.write('\n\n')
        file.write(f'Precision Score: Temporal={tmp_pscore}, Cause={cause_pscore}\n')
        file.write(f'Recall Score: Temporal={tmp_rscore}, Cause={cause_rscore}\n')
        file.write(f'F1 Score: Temporal={tmp_fscore}, Cause={cause_fscore}\n')
        file.write('\n\n')
        file.write('Most tmp unagreed mentions:\n')
        for mention, count in dict(sorted(tmp_unagreed.items(), key=lambda item: item[1], reverse=True)).items():
            file.write(f'{mention})={count}\n')

    print(f"DataFrame written to {file_path}")


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


def main():
    with open(gold_file) as f:
        gold = json.load(f)

    with open(predictions_file) as f:
        predictions = json.load(f)

    print('Gold File')
    gold_ment, _, _ = count_stats_in_file(gold)
    print('Predictions File')
    pred_ment, _, _ = count_stats_in_file(predictions)

    create_report(gold_ment, gold["allPairs"], pred_ment, predictions["allPairs"])


if __name__ == "__main__":
    output_file = f'data/my_data/output/59d4_evaluation_v2'
    gold_file = f'data/my_data/input/59d4_tmp_benji.json'
    predictions_file = f'data/my_data/input/59d4_tmp_michael.json'
    main()
