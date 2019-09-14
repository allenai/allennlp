# pylint: disable=no-self-use,invalid-name,protected-access
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import Hotflip
from allennlp.interpret.attackers.hotflip import DEFAULT_IGNORE_TOKENS

from ..modules.language_model_heads.linear import LinearLanguageModelHead  # pylint: disable=unused-import


class TestHotflip(AllenNlpTestCase):
    def test_hotflip(self):
        inputs = {
                "premise": "I always write unit tests for my code.",
                "hypothesis": "One time I didn't write any unit tests for my code."
        }

        archive = load_archive(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'textual-entailment')

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, 'hypothesis', 'grad_input_1')
        assert attack is not None
        assert 'final' in attack
        assert 'original' in attack
        assert 'outputs' in attack
        assert len(attack['final'][0]) == len(attack['original']) # hotflip replaces words without removing

        # test using SQuAD model (tests different equals method)
        inputs = {
                "question": "OMG, I heard you coded a test that succeeded on its first attempt, is that true?",
                "passage": "Bro, never doubt a coding wizard! I am the king of software, MWAHAHAHA"
        }

        archive = load_archive(self.FIXTURES_ROOT / 'bidaf' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'machine-comprehension')

        hotflipper = Hotflip(predictor)
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs,
                                             'question',
                                             'grad_input_2')
        print(attack)
        assert attack is not None
        assert 'final' in attack
        assert 'original' in attack
        assert 'outputs' in attack
        assert len(attack['final'][0]) == len(attack['original']) # hotflip replaces words without removing

        instance = predictor._json_to_instance(inputs)
        assert instance['question'] != attack['final'][0] # check that the input has changed.

        outputs = predictor._model.forward_on_instance(instance)
        original_labeled_instance = predictor.predictions_to_labeled_instances(instance, outputs)[0]
        original_span_start = original_labeled_instance['span_start'].sequence_index
        original_span_end = original_labeled_instance['span_end'].sequence_index

        flipped_span_start = attack['outputs']['best_span'][0]
        flipped_span_end = attack['outputs']['best_span'][1]

        for i, token in enumerate(instance['question']):
            token = str(token)
            if token in DEFAULT_IGNORE_TOKENS:
                assert token in attack['final'][0] # ignore tokens should not be changed
            # HotFlip keeps changing tokens until either the prediction changes or all tokens have
            # been changed. If there are tokens in the HotFlip final result that were in the
            # original (i.e., not all tokens were flipped), then the prediction should be
            # different.
            else:
                if token == attack['final'][0][i]:
                    assert original_span_start != flipped_span_start or original_span_end != flipped_span_end

    def test_targeted_attack_from_json(self):
        inputs = {"sentence": "The doctor ran to the emergency room to see [MASK] patient."}

        archive = load_archive(self.FIXTURES_ROOT / 'masked_language_model' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'masked_language_model')

        hotflipper = Hotflip(predictor, vocab_namespace='tokens')
        hotflipper.initialize()
        attack = hotflipper.attack_from_json(inputs, target={'words': ['hi']})
        assert attack is not None
        assert 'final' in attack
        assert 'original' in attack
        assert 'outputs' in attack
        assert len(attack['final'][0]) == len(attack['original']) # hotflip replaces words without removing
        assert attack['final'][0] != attack['original']
