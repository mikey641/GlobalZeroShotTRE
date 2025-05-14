from scripts.eval.shared.evaluation import evaluation
from scripts.utils.classes.datasets_type import NarrativeDataset, MATRES_DATASET_NAME, EventFullDataset, MatresDataset, \
    TBDDataset
from scripts.utils.io_utils import Event_Rel, read_file, read_pred_dot_file


def convert_format(orig_ins_list, pred_as_dict, labels, debug=False):
    pred_for_trans = {}
    gold_for_trans = {}
    all_golds = []
    all_preds = []
    count_nas = 0

    # Consider only the gold labels (for all_golds and all_preds)
    for gold_ins in orig_ins_list:
        if gold_ins.target == 'included':
            gold_ins.target = 'is_included'
        elif gold_ins.target == 'include':
            gold_ins.target = 'includes'

        all_golds.append(labels[gold_ins.label])

        if gold_ins.docid not in pred_for_trans:
            pred_for_trans[gold_ins.docid] = []
            gold_for_trans[gold_ins.docid] = []

        key = f'{gold_ins.docid}#{gold_ins.source}#{gold_ins.target}'
        rev_key = f'{gold_ins.docid}#{gold_ins.target}#{gold_ins.source}'
        set_label = -1
        if key in pred_as_dict:
            set_label = pred_as_dict[key]
            all_preds.append(set_label)
        elif rev_key in pred_as_dict:
            set_label = labels.get_reverse_numerical_label(pred_as_dict[rev_key])
            all_preds.append(set_label)
        else:
            all_preds.append(0)
            count_nas += 1
            print(f"NA: {gold_ins.docid}#{gold_ins.source}#{gold_ins.target}")

        gold_for_trans[gold_ins.docid].append((gold_ins.source, labels[gold_ins.label], gold_ins.target))

        if debug:
            if labels[gold_ins.label] != set_label:
                print(f'relation-{key}, gold={labels[gold_ins.label]}, pred={set_label}')

    # Consider all predictions to generate the pred_for_trans obj
    for key, pred in pred_as_dict.items():
        split = key.split('#')
        pred_for_trans[split[0]].append((split[1], pred, split[2]))

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

    print('-' * 50)
    for doc_id in doc_wise_preds:
        all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(orig_doc_wise[doc_id], doc_wise_preds[doc_id], labels, debug=False)
        doc_f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, dataset_type, print_confusion=True)
        print(f"DocID: {doc_id}: F1: {doc_f1}")


if __name__ == "__main__":
    # \\"[a-z]*\(13\)\\" -- \\"[a-z]*\(20\)\\"
    _prediction_file = "data/my_data/prompt/new_expr/omnitemp/omni_DeepSeek-R1_task_description_4res_only_timeline_2.json"
    _dataset_type = EventFullDataset()

    _test_docs_dict, _orig_ins_list = read_file(_dataset_type.get_test_file())
    _labels = _dataset_type.get_label_set()
    _dataset_name = _dataset_type.get_name()

    if _dataset_name == MATRES_DATASET_NAME:
        matres_conversion(_orig_ins_list)

    _pred_as_dict, _ = read_pred_dot_file(_prediction_file, _test_docs_dict, _dataset_type)

    _all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _count_nas = convert_format(_orig_ins_list, _pred_as_dict, _labels, debug=False)

    doc_wise_eval(_pred_as_dict, _orig_ins_list, _labels, _dataset_type)

    evaluation(_all_golds, _all_preds, _gold_for_trans, _pred_for_trans, _dataset_type)

    print()
    print(f"Number of NAs: {_count_nas}")
    count_pred = {i:_all_preds.count(i) for i in _all_preds}
    count_gold = {i:_all_golds.count(i) for i in _all_golds}
    print(f"Predictions dist: {dict(count_pred)}")
    print(f"Gold dist: {dict(count_gold)}")
    print("Done!")