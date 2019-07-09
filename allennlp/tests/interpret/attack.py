# pylint: disable=no-self-use,invalid-name
from pytest import approx
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import InputReduction, Hotflip

class TestAttack(AllenNlpTestCase):
    def test_attackers(self):
        inputs = {
            "premise": "I always write unit tests for my code.",
            "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')
        result = predictor.predict_json(inputs)

        # Hotflip
        hotflipper = Hotflip(predictor)
        attack = hotflipper.attack_from_json(inputs, 'hypothesis', 'grad_input_1')                
        assert attack is not None        
        assert 'final' in attack
        assert 'original' in attack 
        assert 'new_prediction' in attack             
        assert len(attack['final'][0]) == len(attack['original']) # hotflip replaces words without removing

        # Input Reduction
        reducer = InputReduction(predictor)
        reduced = reducer.attack_from_json(inputs, 'hypothesis', 'grad_input_1')                        
        assert reduced is not None
        assert 'final' in reduced
        assert 'original' in reduced         
        assert len(reduced['final'][0]) <= len(reduced['original']) # input reduction removes tokens
        for word in reduced['final'][0]: # no new words entered
            assert word in reduced['original']