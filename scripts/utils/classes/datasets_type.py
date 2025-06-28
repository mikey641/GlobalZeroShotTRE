from scripts.utils.classes.label_sets import FourRelsLabels, SixRelsLabels

MATRES_DATASET_NAME = "matres"
EVENTFULL_DATASET_NAME = "eventfull"
TBD_DATASET_NAME = "tbd"
NARRATIVE_DATASET_NAME = "nt"
NARRATIVE_4RELS_DATASET_NAME = "nt4rels"
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
        elif name == EVENTFULL_DATASET_NAME:
            return EventFullDataset()
        elif name == TBD_DATASET_NAME:
            return TBDDataset()
        elif name == NARRATIVE_DATASET_NAME:
            return NarrativeDataset()
        elif name == NARRATIVE_4RELS_DATASET_NAME:
            return NarrativeDataset4Rels()
        else:
            raise ValueError(f"Unknown dataset name: {name}")


class MatresDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), MATRES_DATASET_NAME, 'data/bayesian_format/testset-temprel.xml')
        # super().__init__(FourRelsLabels(), MATRES_DATASET_NAME, 'data/testset_20events_matres.xml')
        # super().__init__(FourRelsLabels(), MATRES_DATASET_NAME, 'data/testset_20more_matres.xml')


class EventFullDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), EVENTFULL_DATASET_NAME, 'data/bayesian_format/testset_eventfull.xml')


class TBDDataset(DataType):
    def __init__(self):
        super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/bayesian_format/testset_tbd.xml')
        # super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/testset_AllPairs_tbd.xml')
        # super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/testset_small_size_tbd.xml')
        # super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/testset_30more_tbd.xml')


class NarrativeDataset(DataType):
    def __init__(self):
        # super().__init__(SixRelsLabels(), NARRATIVE_DATASET_NAME, 'data/bayesian_format/testset_nt.xml')
        super().__init__(SixRelsLabels(), NARRATIVE_DATASET_NAME, 'data/bayesian_format/testset_nt_50.xml')
        # super().__init__(SixRelsLabels(), NARRATIVE_DATASET_NAME, 'data/testset_consSent_nt.xml')
        # super().__init__(SixRelsLabels(), TBD_DATASET_NAME, 'data/testset_small_size_tbd.xml')


class NarrativeDataset4Rels(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), NARRATIVE_4RELS_DATASET_NAME, 'data/bayesian_format/testset_nt4rels.xml')


class MAVENDataset(DataType):
    def __init__(self):
        super().__init__(FourRelsLabels(), MAVEN_DATASET_NAME, 'data/bayesian_format/validset_maven.xml')
