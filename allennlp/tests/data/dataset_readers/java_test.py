# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import JavaDatasetReader


def assert_dataset_correct(dataset):
    instances = list(dataset)
    assert len(instances) == 10

class JavaDatasetReaderTest(AllenNlpTestCase):
    def test_reader_reads(self):
        reader = JavaDatasetReader.from_params(Params({
            "utterance_indexers": {"namespace": "utterance"},
            "min_identifier_count": 3,
            "num_dataset_instances": -1,
            "linking_feature_extractors": [
                "exact_token_match",
                "contains_exact_token_match",
                "edit_distance",
                "span_overlap_fraction"
            ]
        }))
        dataset = reader.read(str(self.FIXTURES_ROOT / "data" / "java" / "sample_data_prototypes.json"))
        assert_dataset_correct(dataset)