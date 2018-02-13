# pylint: disable=no-self-use,invalid-name

from allennlp.common.testing.model_test_case import ModelTestCase

class SpanConstituencyParserTest(ModelTestCase):

    def setUp(self):
        super(SpanConstituencyParserTest, self).setUp()
        self.set_up_model("tests/fixtures/constituency_parser/constituency_parser.json",
                          "tests/fixtures/data/example_ptb.trees")

    def test_span_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_decode_runs(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.decode(output_dict)

        print(decode_output_dict)