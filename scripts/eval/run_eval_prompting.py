from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import MATRES_DATASET_NAME, OmniTempDataset
from scripts.utils.io_utils import Event_Rel, read_file, read_pred_dot_file, load_golds


def convert_format(test_as_dict, pred_as_dict, labels, debug=False):
    pred_for_trans = {}
    gold_for_trans = {}
    all_golds = []
    all_preds = []
    count_nas = 0

    # Consider only the gold labels (for all_golds and all_preds)
    for key, gold_label in test_as_dict.items():
        all_golds.append(gold_label)
        doc_id, source, target = key.split('#')
        if doc_id not in pred_for_trans:
            pred_for_trans[doc_id] = []
            gold_for_trans[doc_id] = []

        rev_key = f'{doc_id}#{target}#{source}'
        pred_label = -1
        if key in pred_as_dict:
            pred_label = pred_as_dict[key]
            pred_for_trans[doc_id].append((source, pred_label, target))
            all_preds.append(pred_label)
        elif rev_key in pred_as_dict:
            pred_label = labels.get_reverse_numerical_label(pred_as_dict[rev_key])
            pred_for_trans[doc_id].append((source, pred_label, target))
            all_preds.append(pred_label)
        else:
            # Setting to before is relation is not found in predictions
            pred_for_trans[doc_id].append((source, 0, target))
            all_preds.append(0)
            count_nas += 1
            print(f"NA: {key}")

        gold_for_trans[doc_id].append((source, gold_label, target))

        if debug:
            if gold_label != pred_label:
                print(f'relation-{key}, gold={gold_label}, pred={pred_label}')


    return all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas


def matres_conversion(orig_ins_list):
    for i in range(len(orig_ins_list)):
        ins = orig_ins_list[i]
        new_ins = Event_Rel(docid=ins.docid.removesuffix(".json"), label=ins.label, source=ins.source.replace("E", ""), target=ins.target.replace("E", ""),
                            token=ins.token, event_ix=ins.event_ix, verbs=ins.verbs, lemma=ins.lemma,
                            part_of_speech=ins.part_of_speech, position=ins.position, length=ins.length,
                            sentdiff=ins.sentdiff)
        orig_ins_list[i] = new_ins


def doc_wise_eval(pred_as_dict, orig_ins_list, labels, dataset_type):
    doc_wise_preds = dict()
    for key, pred in pred_as_dict.items():
        doc_id = key.split('#')[0]
        if doc_id not in doc_wise_preds:
            doc_wise_preds[doc_id] = dict()
        doc_wise_preds[doc_id][key] = pred

    orig_doc_wise = dict()
    for ins in orig_ins_list:
        if ins.docid not in orig_doc_wise:
            orig_doc_wise[ins.docid] = list()
        orig_doc_wise[ins.docid].append(ins)

    doc_preds = dict()
    for doc_id in doc_wise_preds:
        all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(orig_doc_wise[doc_id], doc_wise_preds[doc_id], labels, debug=False)
        doc_f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type, print_confusion=True)
        doc_preds[doc_id] = doc_f1

    return doc_preds


def eval_sent_diff(pred_as_dict, orig_ins_list, labels, dataset_type, consecutive):
    final_ins_list = []
    final_pred_dict = {}
    for ins in orig_ins_list:
        doc_id = ins.docid
        source = ins.source
        target = ins.target
        sentdiff = ins.sentdiff
        key = f'{doc_id}#{source}#{target}'
        rev_key = f'{doc_id}#{target}#{source}'

        if (consecutive and sentdiff <= 1) or (not consecutive and sentdiff > 1):
            final_ins_list.append(ins)
            if key in pred_as_dict:
                final_pred_dict[key] = pred_as_dict[key]
            elif rev_key in pred_as_dict:
                final_pred_dict[key] = labels.get_reverse_numerical_label(_pred_as_dict[rev_key])

    all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(final_ins_list, final_pred_dict, labels, debug=False)
    return evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type)


if __name__ == "__main__":
    # \\"[a-z]*\(13\)\\" -- \\"[a-z]*\(20\)\\"
    _prediction_file = "output/prompt_OnlyTimeLine_eventfull_gpt4o_task_description_1.json"
    _dataset_type = OmniTempDataset()

    _test_as_dict, _all_test_files = load_golds(_dataset_type.get_test_file(), _dataset_type.get_label_set())
    _labels = _dataset_type.get_label_set()
    _dataset_name = _dataset_type.get_name()

    _pred_as_dict, _ = read_pred_dot_file(_prediction_file, _all_test_files, _dataset_type)

    _all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _count_nas = convert_format(_test_as_dict, _pred_as_dict, _labels, debug=False)

    print('\n\n####### Full Document Evaluation ####')
    f1_full = evaluation(_all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _dataset_type)

    print('\n\n###### Summary ######')
    print(f"Number of NAs: {_count_nas}")
    count_pred = {i:_all_preds.count(i) for i in _all_preds}
    count_gold = {i:_all_golds.count(i) for i in _all_golds}
    print(f"Predictions dist: {dict(count_pred)}")
    print(f"Gold dist: {dict(count_gold)}")
    print(f"Full F1: {f1_full}")

    print("Done!")
