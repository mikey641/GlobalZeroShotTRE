import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, \
    precision_score

from scripts.eval.dataset.utils import parse_DOT
from scripts.eval.model.compute_metrics import calculate
from scripts.utils.check_trans import count_discrepancies
from scripts.utils.classes.datasets_type import EVENTFULL_DATASET_NAME, MATRES_DATASET_NAME, \
    NARRATIVE_4RELS_DATASET_NAME, TBD_DATASET_NAME, NARRATIVE_DATASET_NAME, MAVEN_DATASET_NAME, TCR_HEB_DATASET_NAME


def confusion2prf(confusion):
    tp = 1.0 * np.sum([confusion[i][i] for i in range(3)])
    if tp == 0.:
        return 0., 0., 0.

    prec = tp / (np.sum(confusion[:4,:3]))
    rec = tp / (np.sum(confusion[:3,:4]))
    f1 = 2.0 / (1.0 / prec + 1.0 / rec)
    return prec,rec,f1


def confusion2prf6class(confusion):
    tp = 1.0 * np.sum([confusion[i][i] for i in range(5)])

    if tp == 0.:
        return 0., 0., 0.

    prec = tp / (np.sum(confusion[:6, :5]))  # Sum of first 6 rows and first 5 columns
    rec = tp / (np.sum(confusion[:5, :6]))  # Sum of first 5 rows and all 6 columns
    f1 = 2.0 / (1.0 / prec + 1.0 / rec)

    return prec, rec, f1

def evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type, print_confusion=True):
    dataset_name = dataset_type.get_name()
    if pred_for_trans is not None and gold_for_trans is not None:
        count_discrepancies(pred_for_trans, gold_for_trans, dataset_type, True)

    report, cl_report, f1 = None, None, -1
    acc = accuracy_score(all_golds, all_preds)
    confu = confusion_matrix(all_golds, all_preds)
    if dataset_name in [MATRES_DATASET_NAME, EVENTFULL_DATASET_NAME, NARRATIVE_4RELS_DATASET_NAME, MAVEN_DATASET_NAME, TCR_HEB_DATASET_NAME]:
        cl_report = classification_report(all_golds, all_preds, digits=4)
        prec, rec, f1 = confusion2prf(confu)
        micro_f1 = f1_score(all_golds, all_preds, average='micro')
        report = "Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f, MICRO_F1=%.4f" % (prec, rec, f1, acc, micro_f1)
    elif dataset_name in [NARRATIVE_DATASET_NAME, TBD_DATASET_NAME]:
        f1_micro = f1_score(all_golds, all_preds, average='micro')
        recall = recall_score(all_golds, all_preds, average='micro')
        precision = precision_score(all_golds, all_preds, average='micro')
        cl_report = classification_report(all_golds, all_preds, digits=4)

        try:
            prec, rec, f1 = confusion2prf6class(confu)
        except:
            prec, rec, f1 = confusion2prf(confu)

        report = "Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f, MICRO_F1=%.4f" % (prec, rec, f1, acc, f1_micro)
        # report = "Prec=%.4f, Rec=%.4f, F1=%.4f, Acc=%.4f, MATRES_F1=%.4f" % (precision, recall, f1_micro, acc, f1)

    if print_confusion:
        print(cl_report)
        print(confu, flush=True)
        print(report)
    return f1


if __name__ == "__main__":
    # prediction_file = "data/my_data/density/100per_gpt4o_1exmp_task_description_v2.json"
    prediction_file = "data/my_data/predictions/new_expr/OmniTemp_gemini-2.0-flash_task_description_4res_only_timeline.json"
    gold_file = "data/DOT_format/EventFull_all_dot.json"
    # gold_file = "data/DOT_format/MATRES_test_dot.json"

    with open(prediction_file) as f:
        predictions = json.load(f)

    with open(gold_file) as f:
        golds = json.load(f)

    if len(golds) != len(predictions):
        print("Error: Number of files in gold and prediction are different")

    _all_predictions = dict()
    for file in predictions.keys():
        predicted_graph = parse_DOT(predictions[file]['target'])
        if predicted_graph is None:
            continue

        gold_graph = parse_DOT(golds[file]['target'])
        _all_predictions[file] = {"generated": predicted_graph, "gold": gold_graph}

    calculate(_all_predictions)
    print("Done!")
