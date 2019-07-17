# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import InputReduction

class TestInputReduction(AllenNlpTestCase):
    def test_input_reduction(self):
        # test using entailment model
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')

        reducer = InputReduction(predictor)
        reduced = reducer.attack_from_json(inputs, 'hypothesis', 'grad_input_1')
        assert reduced is not None
        assert 'final' in reduced
        assert 'original' in reduced
        assert reduced['final'][0] # always at least one token
        assert len(reduced['final'][0]) <= len(reduced['original']) # input reduction removes tokens
        for word in reduced['final'][0]: # no new words entered
            assert word in reduced['original']

        # test using NER model (tests different underlying logic)
        inputs = {
                "sentence": "Eric Wallace was an intern at AI2",
        }

        archive = load_archive(self.FIXTURES_ROOT / 'simple_tagger' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'sentence-tagger')

        reducer = InputReduction(predictor)
        reduced = reducer.attack_from_json(inputs, 'tokens', 'grad_input_1')
        assert reduced is not None
        assert 'final' in reduced
        assert 'original' in reduced
        for reduced_input in reduced['final']:
            assert reduced_input # always at least one token
            assert len(reduced_input) <= len(reduced['original']) # input reduction removes tokens
            for word in reduced_input: # no new words entered
                assert word in reduced['original']
