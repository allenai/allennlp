# pylint: disable=no-self-use,invalid-name,no-value-for-parameter

import torch

from allennlp.common.testing.model_test_case import ModelTestCase

class BiaffineDependencyParserTest(ModelTestCase):

    def setUp(self):
        super(BiaffineDependencyParserTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / "biaffine_dependency_parser" / "experiment.json",
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
                                                      'predicted_dependencies', 'predicted_heads',
                                                      'words', 'pos'])

    def test_mst_respects_no_outgoing_root_edges_constraint(self):
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that j is the head of i". In this
        # case, we have heads pointing to their children.

        # We want to construct a case that has 2 children for the ROOT node,
        # because in a typical dependency parse there should only be one
        # word which has the ROOT as it's head.

        energy = torch.Tensor([[0, 2, 1],
                               [10, 0, 0.5],
                               [9, 0.2, 0]]).view(1, 1, 3, 3)
        # In this case, the maximum weight tree looks like this,
        # with both edges having weight 10.
        #               A(index 0)
        #               /        \
        #              \/        \/
        #          B(index 1)   C(index 2)
        length = torch.LongTensor([3])
        heads, _ = self.model._run_mst_decoding(energy, length) # pylint: disable=protected-access
        # This is the correct MST, but not desirable for dependency parsing.
        assert heads.tolist()[0] == [-1, 0, 0]

        energy[:, :, 0, :] = 0
        heads, _ = self.model._run_mst_decoding(energy, length) # pylint: disable=protected-access
        assert heads.tolist()[0] == [-1, 0, 1]
