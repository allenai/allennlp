

from allennlp.models.simple_tagger import SimpleTagger
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary


class SimpleTaggerTest(AllenNlpTestCase):

    def setUp(self):
        self.write_sequence_tagging_files()


    def test_simple_tagger_can_train(self):

        vocab =
        model = SimpleTagger()


        trainer =


