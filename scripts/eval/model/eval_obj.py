from typing import List, Tuple, Dict


class EvalObj:
    def __init__(self, edge_list):
        self.orig_edge_list = edge_list
        self.node_set = set()
        self.edge_set = set()
        self.fill_all_edges()

        self.orig_distribution = EvalObj.calc_edge_distributions(self.orig_edge_list)
        self.set_distribution = EvalObj.calc_edge_distributions(self.edge_set)

    def fill_all_edges(self):
        for edge in self.orig_edge_list:
            self.node_set.add(edge[0])
            self.node_set.add(edge[2])

        self.fill_transitive_closure()

    def fill_transitive_closure(self):
        node_list = sorted(list(self.node_set))
        graph_matrix, index_map = self.get_direct_reach_graph(node_list)
        length = len(graph_matrix)

        for k in range(length):
            for i in range(length):
                for j in range(length):
                    # skip self relations
                    if i == j:
                        continue

                    inferred_rel = EvalObj.get_transitive_relation(graph_matrix, i, j, k)
                    if inferred_rel == 'before':
                        graph_matrix[i][j] = 'before'
                        EvalObj.set_edge_in_graph(graph_matrix, i, j, 'after')
                    elif inferred_rel == 'after':
                        graph_matrix[i][j] = 'after'
                        EvalObj.set_edge_in_graph(graph_matrix, i, j, 'before')
                    elif inferred_rel == 'equal':
                        graph_matrix[i][j] = 'equal'
                        EvalObj.set_edge_in_graph(graph_matrix, i, j, 'equal')

        self.from_matrix_to_edges(graph_matrix, index_map)

    def get_direct_reach_graph(self, node_list: List):
        graph_matrix = [["NA" for _ in range(len(node_list))] for _ in range(len(node_list))]
        index_map = {node: i for i, node in enumerate(node_list)}

        # Fill the direct edges
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            graph_matrix[index_map[e1]][index_map[e2]] = rel

        # Fill the missing symmetric edges
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            if rel == "equal":
                EvalObj.set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "equal")
            elif rel == "before":
                EvalObj.set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "after")
            elif rel == "after":
                EvalObj.set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "before")
            elif rel == 'vague':
                EvalObj.set_edge_in_graph(graph_matrix, index_map[e1], index_map[e2], "vague")

        return graph_matrix, index_map

    @staticmethod
    def get_transitive_relation(reach_graph, i, j, k):
        if ((reach_graph[i][k] == 'after' and reach_graph[k][j] == 'after') or
                (reach_graph[i][k] == 'after' and reach_graph[k][j] == 'equal') or
                (reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'after')):
            return 'after'
        elif ((reach_graph[i][k] == 'before' and reach_graph[k][j] == 'before') or
              (reach_graph[i][k] == 'before' and reach_graph[k][j] == 'equal') or
              (reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'before')):
            return 'before'
        elif reach_graph[i][k] == 'equal' and reach_graph[k][j] == 'equal':
            return 'equal'
        else:
            return 'NA'

    @staticmethod
    def set_edge_in_graph(graph_matrix, i, j, excepted_symmetric_rel):
        if graph_matrix[j][i] == "NA":
            graph_matrix[j][i] = excepted_symmetric_rel
        elif graph_matrix[j][i] != excepted_symmetric_rel:
            graph_matrix[i][j] = "contradict"
            graph_matrix[j][i] = "contradict"

    def from_matrix_to_edges(self, graph_matrix, index_map):
        reversed_index_map = {v: k for k, v in index_map.items()}
        for i in range(len(graph_matrix)):
            for j in range(i + 1, len(graph_matrix)):
                if graph_matrix[i][j] != "NA":
                    self.edge_set.add((reversed_index_map[i], graph_matrix[i][j], reversed_index_map[j]))

    @staticmethod
    def calc_edge_distributions(edges):
        ret_dist = dict()
        for edge in edges:
            e1, rel, e2 = edge
            ret_dist[rel] = ret_dist.get(rel, 0) + 1

        return ret_dist

    def get_edge_map(self):
        return {f'{edge[0]}#{edge[2]}' for edge in self.edge_set}

    def filter_edges(self, consider_edges):
        new_set = set(filter(lambda x: self.filter_method(x, consider_edges), self.edge_set))
        self.set_distribution = EvalObj.calc_edge_distributions(new_set)
        removed = len(self.edge_set) - len(new_set)
        self.edge_set = new_set
        return removed

    def filter_method(self, edge, consider_edges):
        e1, rel, e2 = edge
        # if there is such an edge (regardless of the relation) is in the gold graph, consider it
        if f'{e1}#{e2}' in consider_edges:
            return True
        # if the edge is not, but one of the nodes is not in the gold graph, we should consider it to punish the model
        elif e1 not in self.node_set or e2 not in self.node_set:
            return True
        return False
