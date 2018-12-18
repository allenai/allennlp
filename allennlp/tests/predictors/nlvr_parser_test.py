# pylint: disable=no-self-use,invalid-name
import json
import os

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestNlvrParserPredictor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.inputs = {'worlds': [[[{'y_loc': 80, 'type': 'triangle', 'color': '#0099ff', 'x_loc': 80,
                                     'size': 20}],
                                   [{'y_loc': 80, 'type': 'square', 'color': 'Yellow', 'x_loc': 13,
                                     'size': 20}],
                                   [{'y_loc': 67, 'type': 'triangle', 'color': 'Yellow', 'x_loc': 35,
                                     'size': 10}]],
                                  [[{'y_loc': 8, 'type': 'square', 'color': 'Yellow', 'x_loc': 57,
                                     'size': 30}],
                                   [{'y_loc': 43, 'type': 'square', 'color': '#0099ff', 'x_loc': 70,
                                     'size': 30}],
                                   [{'y_loc': 59, 'type': 'square', 'color': 'Yellow', 'x_loc': 47,
                                     'size': 10}]]],
                       'identifier': 'fake_id',
                       'sentence': 'Each grey box contains atleast one yellow object touching the edge'}

    def test_predictor_with_coverage_parser(self):
        archive_dir = self.FIXTURES_ROOT / 'semantic_parsing' / 'nlvr_coverage_semantic_parser' / 'serialization'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'nlvr-parser')

        result = predictor.predict_json(self.inputs)
        assert 'logical_form' in result
        assert 'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result['denotations'][0]) == 2  # Because there are two worlds in the input.

    def test_predictor_with_direct_parser(self):
        archive_dir = self.FIXTURES_ROOT / 'semantic_parsing' / 'nlvr_direct_semantic_parser' / 'serialization'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'nlvr-parser')

        result = predictor.predict_json(self.inputs)
        assert 'logical_form' in result
        assert 'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result['denotations'][0]) == 2  # Because there are two worlds in the input.

    def test_predictor_with_string_input(self):
        archive_dir = self.FIXTURES_ROOT / 'semantic_parsing' / 'nlvr_coverage_semantic_parser' / 'serialization'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'nlvr-parser')

        self.inputs['worlds'] = json.dumps(self.inputs['worlds'])
        result = predictor.predict_json(self.inputs)
        assert 'logical_form' in result
        assert 'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result['denotations'][0]) == 2  # Because there are two worlds in the input.

    def test_predictor_with_single_world(self):
        archive_dir = self.FIXTURES_ROOT / 'semantic_parsing' / 'nlvr_coverage_semantic_parser' / 'serialization'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'nlvr-parser')

        self.inputs['structured_rep'] = self.inputs['worlds'][0]
        del self.inputs['worlds']
        result = predictor.predict_json(self.inputs)
        assert 'logical_form' in result
        assert 'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result['denotations'][0]) == 1  # Because there is one world in the input.

    def test_predictor_with_single_world_and_string_input(self):
        archive_dir = self.FIXTURES_ROOT / 'semantic_parsing' / 'nlvr_coverage_semantic_parser' / 'serialization'
        archive = load_archive(os.path.join(archive_dir, 'model.tar.gz'))
        predictor = Predictor.from_archive(archive, 'nlvr-parser')

        self.inputs['structured_rep'] = json.dumps(self.inputs['worlds'][0])
        del self.inputs['worlds']
        result = predictor.predict_json(self.inputs)
        assert 'logical_form' in result
        assert 'denotations' in result
        # result['denotations'] is a list corresponding to k-best logical forms, where k is 1 by
        # default.
        assert len(result['denotations'][0]) == 1  # Because there is one world in the input.
