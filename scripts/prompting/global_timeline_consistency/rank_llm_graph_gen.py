from scripts.eval.prompt.run_eval_prompting import matres_conversion
from scripts.utils.classes.datasets_type import NarrativeDataset, MATRES_DATASET_NAME
from scripts.utils.io_utils import read_file, read_pred_dot_file

if __name__ == "__main__":
    _prediction_file = "data/my_data/predictions/new_expr/nt/nt_DeepSeek-R1_task_description_6res_only_timeline_0.json"
    _dataset_type = NarrativeDataset()

    _test_docs_dict, _ = read_file(_dataset_type.get_test_file())
    _labels = _dataset_type.get_label_set()
    _dataset_name = _dataset_type.get_name()

    _pred_as_dict, _pred_triplets = read_pred_dot_file(_prediction_file, _test_docs_dict, _dataset_type)
    print()
