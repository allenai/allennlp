# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common.testing import ModelTestCase
from allennlp.semparse.contexts.quarel_utils import group_worlds, to_qr_spec_string


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

    def test_wdp_zeroshot_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_wdp_zeroshot.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_with_tagger_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_w_tagger.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_entity_bits_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_entity_bits.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_friction_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_friction.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_friction_zeroshot_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_parser_friction_zeroshot.json'  # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_denotation_only_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_denotation_only.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    def test_tagger_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / 'semantic_parsing' / 'quarel' / 'experiment_tagger.json'
        self.ensure_model_can_train_save_and_load(param_file, gradients_to_ignore=self.ignore)

    # Misc util function tests
    def test_group_worlds(self):
        tags = ['B-world', 'B-world', 'I-world', 'O', 'B-world', 'I-world', 'B-world']
        tokens = ['woodfloor', 'blue', 'rug', 'in', 'wood', 'floor', 'rug']
        worlds = group_worlds(tags, tokens)
        assert worlds['world1'] == ['wood floor', 'woodfloor']
        assert worlds['world2'] == ['blue rug', 'rug']

    def test_(self):
        qr_spec = [
                {"friction": 1, "speed": -1, "smoothness": -1, "distance": -1, "heat": 1},
                {"speed": 1, "time": -1}
        ]
        qr_spec_string = to_qr_spec_string(qr_spec)
        assert qr_spec_string == '[friction, -speed, -smoothness, -distance, +heat]\n[speed, -time]'
