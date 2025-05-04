from collections import namedtuple, Counter

from numpy.ma.extras import average

from scripts.eval.prompt.eval_global_consistency import run_majority_vote_trans_const
from scripts.utils.classes.datasets_type import MatresDataset, NarrativeDataset, TBDDataset
from scripts.utils.io_utils import Event_Rel, read_file


def convert_format(orig_ins_list, pred_as_dict, labels, debug=False):
    pred_for_trans = {}
    gold_for_trans = {}
    all_golds = []
    all_preds = []
    count_nas = 0
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
            pred_for_trans[gold_ins.docid].append((gold_ins.source, set_label, gold_ins.target))
            all_preds.append(set_label)
        elif rev_key in pred_as_dict:
            set_label = labels.get_reverse_numerical_label(pred_as_dict[rev_key])
            pred_for_trans[gold_ins.docid].append((gold_ins.source, set_label, gold_ins.target))
            all_preds.append(set_label)
        else:
            pred_for_trans[gold_ins.docid].append((gold_ins.source, 0, gold_ins.target))
            all_preds.append(0)
            count_nas += 1
            print(f"NA: {gold_ins.docid}#{gold_ins.source}#{gold_ins.target}")

        gold_for_trans[gold_ins.docid].append((gold_ins.source, labels[gold_ins.label], gold_ins.target))

        if debug:
            if labels[gold_ins.label] != set_label:
                print(f'relation-{key}, gold={labels[gold_ins.label]}, pred={set_label}')


    return all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas


def split_to_docs_matres(test_order_list, buckets):
    split_docs = {}
    for i, doc in enumerate(test_order_list):
        if doc.docid not in split_docs:
            split_docs[doc.docid] = {'nodes': set(), 'edges': list(), 'relations': []}
        # (Source_event, Target_event, Pred_Relation, Gold_Label, Doc_Id)
        split_docs[doc.docid]['nodes'].add(doc.source)
        split_docs[doc.docid]['nodes'].add(doc.target)
        split_docs[doc.docid]['edges'].append((doc.source, doc.target))
        split_docs[doc.docid]['relations'].append(doc.label)

    doc_ids_buckets = []
    for buck in buckets:
        doc_ids_buckets.append([key for key, value in split_docs.items() if len(value['nodes']) <= buck])

    return split_docs, doc_ids_buckets


def split_to_docs_tbd(test_order_list, buckets):
    split_docs = {}
    for i, doc in enumerate(test_order_list):
        if doc.docid not in split_docs:
            split_docs[doc.docid] = {'nodes': set(), 'edges': list(), 'relations': []}
        # (Source_event, Target_event, Pred_Relation, Gold_Label, Doc_Id)
        split_docs[doc.docid]['nodes'].add(doc.source)
        split_docs[doc.docid]['nodes'].add(doc.target)
        split_docs[doc.docid]['edges'].append((doc.source, doc.target))
        split_docs[doc.docid]['relations'].append(doc.label)

    doc_ids_buckets = [ [] for _ in range(len(split_docs)) ]
    for i, (key, value) in enumerate(sorted([(key, value) for key, value in split_docs.items()], key=lambda x: len(x[1]['nodes']))):
        doc_ids_buckets[i].append(key)
        # if i > 0:
        #     doc_ids_buckets[i] = doc_ids_buckets[i-1] + doc_ids_buckets[i]

    return split_docs, doc_ids_buckets


def orig_list_from_keys(keys, orig_list):
    orig_keys_list = []
    for key in keys:
        for ins in orig_list:
            if ins.docid == key:
                orig_keys_list.append(ins)
    return orig_keys_list


def matres_conversion(orig_ins_list):
    for i in range(len(orig_ins_list)):
        ins = orig_ins_list[i]
        new_ins = Event_Rel(docid=ins.docid.removesuffix(".json"), label=ins.label, source=ins.source.replace("E", ""), target=ins.target.replace("E", ""),
                            token=ins.token, event_ix=ins.event_ix, verbs=ins.verbs, lemma=ins.lemma,
                            part_of_speech=ins.part_of_speech, position=ins.position, length=ins.length,
                            sentdiff=ins.sentdiff)
        orig_ins_list[i] = new_ins


if __name__ == "__main__":
    # \\"[a-z]*\(13\)\\" -- \\"[a-z]*\(20\)\\"
    _prediction_files = [
        "data/my_data/prompt/new_expr/tbd_DeepSeek-R1_task_description_6res_only_timeline_0.json",
        "data/my_data/prompt/new_expr/tbd_DeepSeek-R1_task_description_6res_only_timeline_1.json",
        "data/my_data/prompt/new_expr/tbd_DeepSeek-R1_task_description_6res_only_timeline_2.json",
        "data/my_data/prompt/new_expr/tbd_DeepSeek-R1_task_description_6res_only_timeline_3.json",
        "data/my_data/prompt/new_expr/tbd_DeepSeek-R1_task_description_6res_only_timeline_4.json",
    ]

    _dataset_type = TBDDataset()

    if _dataset_type.get_name() == 'matres':
        _buckets = [10, 15, 20, 25, 30, 100]
    else:
        _buckets = [20, 25, 30, 50, 60, 100]


    _test_docs_dict, _orig_ins_list = read_file(_dataset_type.get_test_file())
    _labels = _dataset_type.get_label_set()
    _dataset_name = _dataset_type.get_name()

    _split_docs, _doc_ids_in_buckets = split_to_docs_matres(_orig_ins_list, _buckets)
    results_bucket = [[] for _ in range(len(_doc_ids_in_buckets))]

    for buk_idx, doc_bucket in enumerate(_doc_ids_in_buckets):
        _new_orig = orig_list_from_keys(doc_bucket, _orig_ins_list)

        try:
            _f1 = run_majority_vote_trans_const(_dataset_type, _test_docs_dict, _prediction_files, _new_orig)
            results_bucket[buk_idx].append(_f1)
            print(f"Bucket- {buk_idx}: F1={_f1}, NAs={'NA'}")
        except Exception as e:
            print(f"Error: {e}")
            continue

    print('--------------------------------')
    _dist_bucket = [[] for _ in range(len(_doc_ids_in_buckets))]
    for i, doc_id_list in enumerate(_doc_ids_in_buckets):
        for doc_id in doc_id_list:
            _dist_bucket[i].extend(_split_docs[doc_id]['relations'])

    for i, res in enumerate(results_bucket):
        print(f"Bucket-{i}: {average(res)}: "
              f"Rels Distribution: {sorted(Counter(_dist_bucket[i]).items(), key=lambda x: x[1], reverse=True)}: "
              f"Total Relations: {len(_dist_bucket[i])}")
    print('--------------------------------')

    print("Done!")
