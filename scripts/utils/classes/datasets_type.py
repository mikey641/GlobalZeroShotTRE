from scripts.utils.classes.label_sets import FourRelsLabels, SixRelsLabels

MATRES_DATASET_NAME = "matres"
OMNITEMP_DATASET_NAME = "omni"
TBD_DATASET_NAME = "tbd"
NARRATIVE_DATASET_NAME = "nt"
MAVEN_DATASET_NAME = "maven"


class DataType(object):
    def __init__(self, labels, name, test_file=None):
        self.label_set = labels
        self.name = name
        self.test_file = test_file

    def get_name(self):
        return self.name

    def get_label_set(self):
        return self.label_set

    def get_test_file(self):
        return self.test_file

    @staticmethod
    def get_dataset_by_name(name):
        if name == MATRES_DATASET_NAME:
            return MatresDataset()
        elif name == OMNITEMP_DATASET_NAME:
            return OmniTempDataset()
        elif name == TBD_DATASET_NAME:
            return TBDDataset()
        elif name == NARRATIVE_DATASET_NAME:
            return NarrativeDataset()
        else:
            raise ValueError(f"Unknown dataset name: {name}")


class MatresDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), MATRES_DATASET_NAME, 'data/MATRES/test')


class OmniTempDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), OMNITEMP_DATASET_NAME, 'data/OmniTemp/test')


class TBDDataset(DataType):
    def __init__(self):
        super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/TBD/test')


class NarrativeDataset(DataType):
    def __init__(self):
        super().__init__(SixRelsLabels(), NARRATIVE_DATASET_NAME, 'data/NT_6/test_18ment')


class MavenDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), MAVEN_DATASET_NAME, 'data/bayesian_format/validset_maven.xml')
