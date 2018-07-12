# pylint: disable=no-self-use,invalid-name,no-value-for-parameter

from allennlp.common.testing.model_test_case import ModelTestCase

class BiaffineDependencyParserTest(ModelTestCase):

    def setUp(self):
        super(BiaffineDependencyParserTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / "dependency_parser" / "experiment.json",
                          self.FIXTURES_ROOT / "data" / "dependencies.conllu")

    def test_dependency_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


    def test_mst_decoding_can_run_forward(self):
        self.model.use_mst_decoding_for_validation = True
        self.ensure_model_can_train_save_and_load(self.param_file)


    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_decode_runs(self):
        self.model.eval()
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)

        assert set(decode_output_dict.keys()) == set(['heads', 'head_tags', 'arc_loss',
                                                      'tag_loss', 'loss', 'mask',
                                                      'predicted_dependencies', 'predicted_heads'])
