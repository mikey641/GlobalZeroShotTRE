from copy import deepcopy

EMPTY_REL = 'NA'
CONT_REL = 'contradict'


class EvalObj:
    def __init__(self, doc_id, edge_list):
        self.doc_id = doc_id
        self.orig_edge_list = edge_list
        self.node_set = set()
        self.edge_set = set()
        self.fill_all_edges()

        self.duplicates = 0
        self.contradictions = 0
        self.orig_node_degree = dict()
        self.calc_degree()
        self.calc_duplicates()
        self.orig_distribution = EvalObj.calc_edge_distributions(self.orig_edge_list)
        self.set_distribution = EvalObj.calc_edge_distributions(self.edge_set)

    def fill_all_edges(self):
        for edge in self.orig_edge_list:
            self.node_set.add(edge[0])
            self.node_set.add(edge[2])

        self.fill_transitive_closure()

    def fill_transitive_closure(self):
        # node_list = sorted(list(self.node_set))
        graph_matrix, index_map = self.get_direct_reach_graph()
        trans_matrix = deepcopy(graph_matrix)
        length = len(trans_matrix)

        for k in range(length):
            for i in range(length):
                for j in range(length):
                    direct_rel = graph_matrix[i][j]
                    # skip self relations
                    if i == j or direct_rel != EMPTY_REL:
                        continue

                    trans_rel = trans_matrix[i][j]
                    inferred_rel = EvalObj.get_transitive_relation(trans_matrix, i, j, k)
                    if inferred_rel != EMPTY_REL:
                        if trans_rel == EMPTY_REL:
                            trans_matrix[i][j] = inferred_rel
                        elif trans_rel != EMPTY_REL and inferred_rel != trans_rel:
                            trans_matrix[i][j] = CONT_REL

        for i in range(length):
            for j in range(length):
                if trans_matrix[i][j] != CONT_REL:
                    graph_matrix[i][j] = trans_matrix[i][j]

        self.from_matrix_to_edges(graph_matrix, index_map)

    @staticmethod
    def get_reverse_relation(rel):
        if rel == 'before':
            return 'after'
        elif rel == 'after':
            return 'before'
        else:
            return rel

    def get_direct_reach_graph(self):
        node_list = sorted(list(self.node_set))
        graph_matrix = [[EMPTY_REL for _ in range(len(node_list))] for _ in range(len(node_list))]
        index_map = {node: i for i, node in enumerate(node_list)}

        # Fill the direct edges
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            current_rel = graph_matrix[index_map[e1]][index_map[e2]]
            rev_rel = graph_matrix[index_map[e2]][index_map[e1]]
            if (current_rel == EMPTY_REL or current_rel == rel) and (rev_rel == EMPTY_REL or rev_rel == EvalObj.get_reverse_relation(rel)):
                graph_matrix[index_map[e1]][index_map[e2]] = rel
                graph_matrix[index_map[e2]][index_map[e1]] = EvalObj.get_reverse_relation(rel)
            else:
                graph_matrix[index_map[e1]][index_map[e2]] = CONT_REL
                graph_matrix[index_map[e2]][index_map[e1]] = CONT_REL

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
            return EMPTY_REL

    def from_matrix_to_edges(self, graph_matrix, index_map):
        reversed_index_map = {v: k for k, v in index_map.items()}
        for i in range(len(graph_matrix)):
            for j in range(i + 1, len(graph_matrix)):
                if graph_matrix[i][j] != EMPTY_REL:
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
            elif contr_key in key_set or contr_key_rev in key_set:
                self.contradictions += 1
            else:
                key_set.add(key)
                key_set.add(key_rev)

    def calc_degree(self):
        for edge in self.orig_edge_list:
            e1, rel, e2 = edge
            self.orig_node_degree[e1] = self.orig_node_degree.get(e1, 0) + 1
            self.orig_node_degree[e2] = self.orig_node_degree.get(e2, 0) + 1
