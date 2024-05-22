class AnnotObj:
    def __init__(self, data, mentions, num_pairs, expected_pairs):
        self.data = data
        self.mentions = mentions
        self.num_pairs = num_pairs
        self.expected_pairs = expected_pairs

    def __str__(self):
        return f'data={self.data}, mentions={self.mentions}, num_pairs={self.num_pairs}, expected_pairs={self.expected_pairs}'

