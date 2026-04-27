"""Evaluate a MATRES prediction file (DOT format) against the gold annotations.

Usage (from GlobalZeroShotTRE/):
    PYTHONPATH=. .venv/bin/python scripts/eval/eval_matres.py \
        output/matres_deepseek_global/matres_DeepSeek-R1_task_description_4res_only_global_0.json
"""
import sys

from scripts.utils.classes.datasets_type import MatresDataset
from scripts.utils.io_utils import read_pred_dot_file, load_golds
from scripts.eval.run_eval_prompting import convert_format
from scripts.eval.shared.evaluation import evaluation

if len(sys.argv) < 2:
    print("Usage: eval_matres.py <prediction_file.json>")
    sys.exit(1)

pred_file = sys.argv[1]
ds = MatresDataset()

test_as_dict, all_test_files = load_golds(ds.get_test_file(), ds.get_label_set())
pred_as_dict, _ = read_pred_dot_file(pred_file, all_test_files, ds)
all_golds, all_preds, gold_for_trans, pred_for_trans, count_nas = convert_format(
    test_as_dict, pred_as_dict, ds.get_label_set()
)

print("\n=== Full Evaluation ===")
f1 = evaluation(all_golds, all_preds, gold_for_trans, pred_for_trans, ds)
print(f"NAs: {count_nas}")
print(f"F1: {f1:.4f}")
