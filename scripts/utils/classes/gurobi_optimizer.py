from typing import Type

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from scripts.utils.classes.datasets_type import DataType


class GurobiOptimizer(object):
    def __init__(self, dataset_type: DataType, alpha=-1):
        self.dataset_type = dataset_type
        self.labels = dataset_type.get_label_set()
        self.classes = self.labels.get_index_to_class()
        self.alpha = alpha

        # Create Gurobi model
        self.model = gp.Model('TemporalRelationsOptimization')

    def init_and_run_constraints(self, all_pairs, nodes):
        node_count = len(nodes)
        n_classes = self.labels.get_num_classes()
        node_indices = {node: idx for idx, node in enumerate(nodes)}

        # Define symmetric class indices
        symmetric_class_indices = [self.labels[self.labels.get_reverse_label(key)] for key in self.labels.get_classes()]

        probs_mat = {}
        for pair in all_pairs:
            i = node_indices[pair[0]]
            j = node_indices[pair[1]]
            probs = pair[2]

            if i not in probs_mat:
                probs_mat[i] = {}
            probs_mat[i][j] = probs

        if self.alpha > 0:
            self.handle_contradictions(probs_mat, node_count)

        # Create variables x_{i,j,c}
        x = self.create_varables(node_count, n_classes)

        # Add constraints: For each pair (i, j), the sum over classes must be 1
        self.add_sum_one(x, n_classes, node_count)

        # Add symmetry constraints: x_{i,j,c} = x_{j,i,s(c)}
        self.add_sym(x, node_count, symmetric_class_indices, n_classes)

        # Add transitivity constraints
        self.add_constraints(x, node_count, n_classes)

        # Set the objective function: maximize the total probability of the assigned classes
        objective = gp.quicksum(
            probs_mat[i][j][c] * x[i, j, c]
            for i in range(node_count)
            for j in range(node_count)
            if i != j
            for c in range(n_classes)
        )

        self.model.setObjective(objective, GRB.MAXIMIZE)

        # Optimize the model
        self.model.optimize()

        # Retrieve and print the results
        all_golds = []
        all_preds = []
        pred_gold_mapping = {}
        for i, pair in enumerate(all_pairs):
            i = node_indices[pair[0]]
            j = node_indices[pair[1]]
            gold_lab = self.labels[pair[3][0]] if pair[3][0] != 'NA' else -1
            all_golds.append(gold_lab)
            for c in range(n_classes):
                if x[i, j, c].X > 0.5:
                    all_preds.append(self.labels[self.classes[c]])
                    pred_gold_mapping[(pair[0], pair[1])] = (gold_lab, self.labels[self.classes[c]])

        return all_golds, all_preds, pred_gold_mapping

    def error_analysis(self, preds, golds, pairs, error_mat, classes):
        errors_log = []
        for i, pair in enumerate(pairs):
            error_mat[golds[i]][preds[i]] += 1
            if preds[i] != golds[i]:
                errors_log.append(f"Predicted: {classes[preds[i]]} # Gold: {classes[golds[i]]} # Pair: {pair}")

        return errors_log

    def create_varables(self, node_count, n_classes):
        x = {}
        for i in range(node_count):
            for j in range(node_count):
                if i != j:
                    for c in range(n_classes):
                        x[i, j, c] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{self.classes[c]}")
        return x

    def const_all(self, x, i, j, k):
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] >= x[i, k, self.labels['BEFORE']] + x[k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_before1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] >= x[i, k, self.labels['BEFORE']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_before2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_before3_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] >= x[i, k, self.labels['AFTER']] + x[k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_after1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] >= x[i, k, self.labels['AFTER']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_after2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_after3_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['EQUAL']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_equal1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_eql_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['VAGUE']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_eql_vag2_{i}_{j}_{k}"
        )

    def const_4rels(self, x, i, j, k):
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['BEFORE']] + x[
                k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_bef_vag1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['VAGUE']] + x[
                k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['AFTER']] + x[
                k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_aft_vag1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] >= x[i, k, self.labels['VAGUE']] + x[
                k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_aft_vag2_{i}_{j}_{k}"
        )

    def const_6rels(self, x, i, j, k):
        self.model.addConstr(
            x[i, j, self.labels['INCLUDES']] >= x[i, k, self.labels['INCLUDES']] + x[k, j, self.labels['INCLUDES']] - 1,
            name=f"Transitivity_before1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['INCLUDES']] >= x[i, k, self.labels['INCLUDES']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_before2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['INCLUDES']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['INCLUDES']] - 1,
            name=f"Transitivity_before2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['IS_INCLUDED']] + x[k, j, self.labels['IS_INCLUDED']] - 1,
            name=f"Transitivity_before1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['IS_INCLUDED']] + x[k, j, self.labels['EQUAL']] - 1,
            name=f"Transitivity_before2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['EQUAL']] + x[k, j, self.labels['IS_INCLUDED']] - 1,
            name=f"Transitivity_before2_{i}_{j}_{k}"
        )

        # Adding 6 rels constraints
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] +
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['BEFORE']] + x[k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_bef_vag1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] +
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['VAGUE']] + x[k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] +
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['AFTER']] + x[k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_aft_vag1_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] +
            x[i, j, self.labels['IS_INCLUDED']] >= x[i, k, self.labels['VAGUE']] + x[k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_aft_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['BEFORE']] + x[k, j, self.labels['INCLUDES']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['INCLUDES']] + x[k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['BEFORE']] + x[k, j, self.labels['IS_INCLUDED']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['IS_INCLUDED']] + x[k, j, self.labels['BEFORE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['AFTER']] + x[k, j, self.labels['INCLUDES']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['INCLUDES']] + x[k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['AFTER']] + x[k, j, self.labels['IS_INCLUDED']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['IS_INCLUDED']] + x[k, j, self.labels['AFTER']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[
                i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['INCLUDES']] + x[k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[
                i, j, self.labels['INCLUDES']] >=
            x[i, k, self.labels['VAGUE']] + x[k, j, self.labels['INCLUDES']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[
                i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['IS_INCLUDED']] + x[k, j, self.labels['VAGUE']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )
        self.model.addConstr(
            x[i, j, self.labels['BEFORE']] + x[i, j, self.labels['AFTER']] + x[i, j, self.labels['VAGUE']] + x[
                i, j, self.labels['IS_INCLUDED']] >=
            x[i, k, self.labels['VAGUE']] + x[k, j, self.labels['IS_INCLUDED']] - 1,
            name=f"Transitivity_bef_vag2_{i}_{j}_{k}"
        )

    def add_constraints(self, x, node_count, n_classes):
        for i in range(node_count):
            for j in range(node_count):
                if i != j:
                    for k in range(node_count):
                        if k != i and k != j:
                            self.const_all(x, i, j, k)
                            if n_classes == 4:
                                self.const_4rels(x, i, j, k)
                            if n_classes == 6:
                                self.const_6rels(x, i, j, k)

    def add_sym(self, x, node_count, symmetric_class_indices, n_classes):
        for i in range(node_count):
            for j in range(node_count):
                if i != j:
                    for c in range(n_classes):
                        s_c = symmetric_class_indices[c]
                        self.model.addConstr(
                            x[i, j, c] == x[j, i, s_c],
                            name=f"Symmetry_{i}_{j}_{self.classes[c]}"
                        )

    def add_sum_one(self, x, n_classes, node_count):
        for i in range(node_count):
            for j in range(node_count):
                if i != j:
                    self.model.addConstr(
                        gp.quicksum(x[i, j, c] for c in range(n_classes)) == 1,
                        name=f"OneClass_{i}_{j}"
                    )

    def handle_contradictions(self, probs_mat, node_count):
        # Handle contradictory predictions
        for i in range(node_count):
            for j in range(node_count):
                if i != j and j in probs_mat.get(i, {}):
                    probs_ij = probs_mat[i][j]
                    probs_ji = probs_mat[j][i]
                    probs_ji_swiped = probs_ji.copy()
                    probs_ji_swiped[0], probs_ji_swiped[1] = probs_ji[1], probs_ji[0]  # Swap indices 0 and 1
                    joint_probs = probs_ij + probs_ji_swiped
                    probs_rels_norm = [p / sum(joint_probs) for p in joint_probs]
                    entropy = -sum(p * np.log(p + 1e-9) for p in probs_rels_norm)
                    adjusted_probs = joint_probs.copy()
                    adjusted_probs[VAGUE_IDX] += self.alpha * entropy

                    total = sum(adjusted_probs)
                    adjusted_probs = [p / total for p in adjusted_probs]

                    probs_mat[i][j] = adjusted_probs.copy()
                    probs_mat[j][i] = adjusted_probs.copy()
                    probs_mat[j][i][0], probs_mat[j][i][1] = probs_mat[j][i][1], probs_mat[j][i][0]
