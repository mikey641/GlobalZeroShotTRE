class AbsLabels(object):
    def __init__(self, labels):
        self.labels = labels

    def get_classes(self):
        return self.labels.keys()

    def get_index_to_class(self):
        return {v: k for k, v in self.labels.items()}

    def get_rel(self, label):
        pass

    def get_num_classes(self):
        return len(self.labels)

    def get_reverse_label(self, label):
        pass

    def get_reverse_numerical_label(self, class_idx):
        return self.labels[self.get_reverse_label(self.get_index_to_class()[class_idx])]

    def __getitem__(self, item):
        return self.labels[item]


class FourRelsLabels(AbsLabels):
    def __init__(self):
        labels = {'BEFORE': 0, 'AFTER': 1, 'EQUAL': 2, 'VAGUE': 3}
        super().__init__(labels)

    def get_rel(self, label):
        if label in ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']:
            return label
        else:
            raise ValueError('Invalid label')

    def get_reverse_label(self, label):
        if label == 'BEFORE':
            return 'AFTER'
        elif label == 'AFTER':
            return 'BEFORE'
        else:
            return label

    def adjust_label(self, label):
        if label in ['EQUAL', 'SAME_TIME', 'SIMULTANEOUS', 'DURING']:
            return 'EQUAL'
        elif label in ['BEFORE', 'PRECEDE']:
            return 'BEFORE'
        elif label in ['VAGUE', 'NONE']:
            return 'VAGUE'
        else:
            return label


class SixRelsLabels(AbsLabels):
    def __init__(self):
        labels = {'BEFORE': 0, 'AFTER': 1, 'INCLUDES': 2, 'IS_INCLUDED': 3, 'EQUAL': 4, 'VAGUE': 5}
        super().__init__(labels)

    def get_rel(self, label):
        if label in ['INCLUDES', 'INCLUDE']:
            return 'INCLUDES'
        elif label in ['IS_INCLUDED', 'INCLUDED']:
            return 'IS_INCLUDED'
        elif label in ['EQUAL', 'SAME_TIME', 'SAME', 'SIMULTANEOUS']:
            return 'EQUAL'
        elif label in ['BEFORE', 'AFTER', 'VAGUE']:
            return label
        else:
            raise ValueError('Invalid label')

    def get_reverse_label(self, label):
        if label == 'BEFORE':
            return 'AFTER'
        elif label == 'AFTER':
            return 'BEFORE'
        elif label in ['IS_INCLUDED', 'INCLUDED']:
            return 'INCLUDES'
        elif label in ['INCLUDES', 'INCLUDE']:
            return 'IS_INCLUDED'
        else:
            return label

    def adjust_label(self, label):
        if label in ['INCLUDES', 'INCLUDE', 'PRECEDE']:
            return 'INCLUDES'
        elif label in ['IS_INCLUDED', 'INCLUDED', 'DURING']:
            return 'IS_INCLUDED'
        elif label in ['EQUAL', 'SAME_TIME', 'SAME', 'SIMULTANEOUS']:
            return 'EQUAL'
        elif label in ['BEFORE', 'PRECEDE']:
            return 'BEFORE'
        elif label in ['VAGUE', 'NONE']:
            return 'VAGUE'
        else:
            return label
