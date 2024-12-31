import json

from scripts.eval.dataset.utils import parse_DOT
from scripts.eval.model.compute_metrics import calculate


if __name__ == "__main__":
    # prediction_file = "data/my_data/density/100per_gpt4o_1exmp_task_description_v2.json"
    prediction_file = "data/my_data/batch_req/matresexmp_gpt4o_1exmp_task_description_v2.json"
    gold_file = "data/DOT_format/EventFull_test_dot.json"
    # gold_file = "data/DOT_format/MATRES_test_dot.json"

    with open(prediction_file) as f:
        predictions = json.load(f)

    with open(gold_file) as f:
        golds = json.load(f)

    if len(golds) != len(predictions):
        print("Error: Number of files in gold and prediction are different")

    all_predictions = dict()
    for file in predictions.keys():
        predicted_graph = parse_DOT(predictions[file]['target'])
        if predicted_graph is None:
            continue

        gold_graph = parse_DOT(golds[file]['target'])
        all_predictions[file] = {"generated": predicted_graph, "gold": gold_graph}

    calculate(all_predictions)
    print("Done!")
