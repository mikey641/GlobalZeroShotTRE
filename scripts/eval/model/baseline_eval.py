import json
import random
from itertools import combinations

from scripts.eval.dataset.utils import parse_DOT
from scripts.eval.model.compute_metrics import calculate

if __name__ == "__main__":
    EDGES = ['before', 'after', 'equal', 'vague']
    run_method = "notrandom"

    gold_file = "data/DOT_format/EventFull_test_dot.json"
    # gold_file = "data/DOT_format/MATRES_test_dot.json"

    with open(gold_file) as f:
        golds = json.load(f)

    all_predictions = dict()
    for file in golds.keys():
        gold_graph = parse_DOT(golds[file]['target'])

        node_set = set()
        for edge in gold_graph:
            node_set.add(edge[0])
            node_set.add(edge[2])

        all_pairs = combinations(list(sorted(node_set)), 2)
        predict_graph = set()
        for pair in all_pairs:
            e1, e2 = pair
            if run_method == "random":
                predict_graph.add((e1, random.choice(EDGES), e2))
            else:
                predict_graph.add((e1, "before", e2))

        all_predictions[file] = {"generated": predict_graph, "gold": gold_graph}

    calculate(all_predictions)
    print("Done!")
