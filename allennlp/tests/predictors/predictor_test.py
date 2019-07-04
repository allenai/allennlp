# pylint: disable=no-self-use,invalid-name
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

class TestPredictor(AllenNlpTestCase):
    def test_from_archive_does_not_consume_params(self):
        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        Predictor.from_archive(archive, 'machine-comprehension')

        # If it consumes the params, this will raise an exception
        Predictor.from_archive(archive, 'machine-comprehension')

    def test_loads_correct_dataset_reader(self):
        # The ATIS archive has both training and validation ``DatasetReaders``. The
        # ``keep_if_unparseable`` argument has a different value in each of them
        archive = load_archive(self.FIXTURES_ROOT / 'semantic_parsing' / 'atis' / 'serialization' / 'model.tar.gz')

        predictor = Predictor.from_archive(archive, 'atis-parser')
        assert predictor._dataset_reader._keep_if_unparseable == False

        predictor = Predictor.from_archive(archive, 'atis-parser', dataset_reader_to_load='train')
        assert predictor._dataset_reader._keep_if_unparseable == False

        predictor = Predictor.from_archive(archive, 'atis-parser', dataset_reader_to_load='validation')
        assert predictor._dataset_reader._keep_if_unparseable == True

    def test_fails_without_validation_dataset_reader(self):
        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        with self.assertRaises(ConfigurationError):
            Predictor.from_archive(archive, 'machine-comprehension', dataset_reader_to_load='validation')

    def test_fails_with_unknown_dataset_reader(self):
        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        with self.assertRaises(ConfigurationError):
            Predictor.from_archive(archive, 'machine-comprehension', dataset_reader_to_load='unknown')
