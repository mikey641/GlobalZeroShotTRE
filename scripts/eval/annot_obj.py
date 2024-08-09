class AnnotObj:
    def __init__(self, data, mentions, num_tmp_pairs, num_coref_pairs, num_cause_pairs, expected_pairs):
        self.data = data
        self.mentions = mentions
        self.num_tmp_pairs = num_tmp_pairs
        self.expected_pairs = expected_pairs
        self.num_coref_pairs = num_coref_pairs
        self.num_cause_pairs = num_cause_pairs

    def __str__(self):
        return (f'data={self.data}, mentions={self.mentions}, num_tmp_pairs={self.num_tmp_pairs}, '
                f'num_coref_pairs={self.num_coref_pairs}, num_cause_pairs={self.num_cause_pairs}, '
                f'expected_pairs={self.expected_pairs}')
