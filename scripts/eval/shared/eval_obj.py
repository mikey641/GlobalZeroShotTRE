from scripts.utils.transitive_algos import triplets_to_numpy_graph, transitive_closure_with_relations, \
    transitive_reduction_with_relations

EMPTY_REL = ''


class EvalObj:
    def __init__(self, doc_id, edge_list):
        self.doc_id = doc_id
        self.orig_edge_list = edge_list
        self.node_set = set()
        self.edge_set = set()
        self.edge_set_reduced = set()
        self.fill_all_edges()

        self.duplicates = 0
        self.contradictions = 0
        self.orig_node_degree = dict()
        self.calc_degree()
        self.calc_duplicates()
        self.orig_distribution = EvalObj.calc_edge_distributions(self.orig_edge_list)
        self.set_distribution = EvalObj.calc_edge_distributions(self.edge_set)
        self.reduced_distribution = EvalObj.calc_edge_distributions(self.edge_set_reduced)

    def fill_all_edges(self):
        for edge in self.orig_edge_list:
            self.node_set.add(edge[0])
            self.node_set.add(edge[2])

        graph_matrix, index_map = triplets_to_numpy_graph(self.orig_edge_list)
        trans_matrix = transitive_closure_with_relations(graph_matrix)
        reduced_matrix, _ = transitive_reduction_with_relations(trans_matrix)

        self.edge_set = self.from_matrix_to_edges(trans_matrix, index_map)
        self.edge_set_reduced = self.from_matrix_to_edges(reduced_matrix, index_map)

    @staticmethod
    def get_reverse_relation(rel):
        if rel == 'before':
            return 'after'
        elif rel == 'after':
            return 'before'
        else:
            return rel

    @staticmethod
    def from_matrix_to_edges(graph_matrix, index_map):
        loc_edge_set = set()
        reversed_index_map = {v: k for k, v in index_map.items()}
        rel_convert = {'B': 'before', 'A': 'after', 'E': 'equal', 'V': 'vague', 'C': 'contradiction'}
        for i in range(len(graph_matrix)):
            for j in range(i + 1, len(graph_matrix)):
                if graph_matrix[i][j] != EMPTY_REL:
                    loc_edge_set.add((reversed_index_map[i], rel_convert[graph_matrix[i][j]], reversed_index_map[j]))
        return loc_edge_set

    @staticmethod
    def calc_edge_distributions(edges):
        ret_dist = dict()
        for edge in edges:
            e1, rel, e2 = edge
            ret_dist[rel] = ret_dist.get(rel, 0) + 1

        return ret_dist

    def get_edge_map(self):
        return {f'{edge[0]}#{edge[2]}' for edge in self.edge_set}

    def filter_edges(self, consider_edges, node_set):
        new_set = set(filter(lambda x: EvalObj.filter_method(x, consider_edges, node_set), self.edge_set))
        self.set_distribution = EvalObj.calc_edge_distributions(new_set)
        removed = len(self.edge_set) - len(new_set)
        self.edge_set = new_set
        return removed

    @staticmethod
    def filter_method(edge, consider_edges, node_set):
        e1, rel, e2 = edge
        # if there is such an edge (regardless of the relation) is in the gold graph, consider it
        if f'{e1}#{e2}' in consider_edges:
            return True
        # if the edge is not, but one of the nodes is not in the gold graph, we should consider it to punish the model
        elif e1 not in node_set or e2 not in node_set:
            return True
        return False

    def calc_duplicates(self):
        key_set = set()
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            key = f'{e1}#{rel}#{e2}'
            key_rev = f'{e2}#{EvalObj.get_reverse_relation(rel)}#{e1}'
            contr_key = f'{e2}#{rel}#{e1}'
            contr_key_rev = f'{e1}#{EvalObj.get_reverse_relation(rel)}#{e2}'
            if key in key_set or key_rev in key_set:
                self.duplicates += 1

            if contr_key in key_set or contr_key_rev in key_set:
                self.contradictions += 1

            key_set.add(key)
            key_set.add(key_rev)

    def calc_degree(self):
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            self.orig_node_degree[e1] = self.orig_node_degree.get(e1, 0) + 1
            self.orig_node_degree[e2] = self.orig_node_degree.get(e2, 0) + 1
