# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import ModelTestCase

class QuarelSemanticParserTest(ModelTestCase):

    def setUp(self):

        super(QuarelSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "quarel" / "experiment_parser.json"),
                          str(self.FIXTURES_ROOT / "data" / "quarel.jsonl"))
        # No gradient for these if only one entity type
        self.ignore = {"_entity_type_encoder_embedding.weight", "_entity_type_decoder_embedding.weight"}

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=self.ignore)

    def test_elmo_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_elmo.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_zeroshot_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_zeroshot.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_with_tagger_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_w_tagger.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_entity_bits_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_entity_bits.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_tagger_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_tagger.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)
