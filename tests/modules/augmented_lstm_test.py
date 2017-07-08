
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.testing.test_case import AllenNlpTestCase
import pytest

class TestAugmentedLSTM(AllenNlpTestCase):

    def test_augmented_lstm_completes_forward_pass(self):


    def test_augmented_lstm_throws_error_on_non_packed_sequence_input(self):


    def test_padded_sequences_are_handled_correctly_forwards(self):

    def test_padded_sequences_are_handled_correctly_backwards(self):