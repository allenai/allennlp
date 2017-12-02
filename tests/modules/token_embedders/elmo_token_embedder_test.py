# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import ModelTestCase
from allennlp.nn.util import arrays_to_variables

class TestElmoTokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model('tests/fixtures/elmo/config/characters_token_embedder.json',
                          'tests/fixtures/data/conll2003.txt')

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_elmo_token_embedder_forward_pass_runs_correctly(self):
        training_arrays = self.dataset.as_array_dict()
        output_dict = self.model.forward(**arrays_to_variables(training_arrays))
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}
