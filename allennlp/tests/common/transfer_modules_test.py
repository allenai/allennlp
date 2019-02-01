# pylint: disable=protected-access
import torch

from allennlp.nn import InitializerApplicator
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.common.transfer_modules import TransferModules
from allennlp.modules import FeedForward, TextFieldEmbedder, TimeDistributed
from allennlp.models import Model
from allennlp.models.archival import load_archive

class DecomposableAttentionChanged(Model):
    """
    Decomposable Attention Model that loads most weights from the
    trained entailment model predicting 3 classes, and forms a new model
    which can predicts any number of classed defined by agggregate_feedfowrad.
    Freezing, tuning, re-initializing configs can be set up according to the params.
    """
    def __init__(self, vocab: Vocabulary,
                 transfer_modules: TransferModules,
                 aggregate_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(DecomposableAttentionChanged, self).__init__(vocab)

        # For text_field_embedder and attend_feedforward we use same path in new model as
        # the original trained model but compare_feedforward is completely different.
        # This mapping will happen without any manual setting and will also work if original
        # _compare_feedforward was set inside somewhere as _submodule_1._submodule_2.compare_feedforward.
        self._text_field_embedder = transfer_modules.modules_dict["_text_field_embedder"].module
        self._attend_feedforward = transfer_modules.modules_dict["_attend_feedforward"].module
        self.scratch_compare_feedforward = transfer_modules.modules_dict["_compare_feedforward"].module

        self._aggregate_feedforward = aggregate_feedforward

        transfer_modules.update_model_initializer(self, initializer)
        initializer(self)

class TestTransferModules(AllenNlpTestCase):


    def setUp(self):
        super(TestTransferModules, self).setUp()
        self._model_archive = str(self.FIXTURES_ROOT / 'decomposable_attention' / 'serialization' / 'model.tar.gz')
        self._transfer_modules_params = {
                "archive_path": self._model_archive,
                "modules_config": {
                        "_text_field_embedder": "freeze",      # load pretrained but don't fine-tune further.
                        "_attend_feedforward": "tune",         # load pretrained and fine-tune further.
                        "_compare_feedforward": "reinitialize" # re-initialize and train.
                }
        }


    def test_transfer_modules_construction(self):

        transfer_modules = TransferModules.from_params(Params(self._transfer_modules_params).duplicate())

        # Test transfer modules types
        module_paths = ["_text_field_embedder", "_attend_feedforward", "_compare_feedforward"]
        assert isinstance(transfer_modules.modules_dict["_text_field_embedder"].module, TextFieldEmbedder)
        assert isinstance(transfer_modules.modules_dict["_attend_feedforward"].module, TimeDistributed)
        assert isinstance(transfer_modules.modules_dict["_compare_feedforward"].module, TimeDistributed)

        # Test initialize attribute of TransferInfo
        assert [transfer_modules.modules_dict[name].initialize for name in module_paths] == [False, False, True]

        # Test requires_grad attribute of TransferInfo
        assert [transfer_modules.modules_dict[name].requires_grad for name in module_paths] == [False, True, True]

        # Test requires_grad for parameters in module attribute of TransferInfo
        for parameter in transfer_modules.modules_dict["_text_field_embedder"].module.parameters():
            assert not parameter.requires_grad

        for parameter in transfer_modules.modules_dict["_attend_feedforward"].module.parameters():
            assert parameter.requires_grad

        for parameter in transfer_modules.modules_dict["_compare_feedforward"].module.parameters():
            assert parameter.requires_grad


    def test_transfer_modules_usage(self):

        trained_model = load_archive(self._model_archive).model
        transfer_model_params = {
                "transfer_modules": self._transfer_modules_params,
                "aggregate_feedforward": {
                        "input_dim": 400,
                        "num_layers": 2,
                        "hidden_dims": [200, 3],
                        "activations": ["relu", "linear"],
                        "dropout": [0.2, 0.0]
                }
        }
        transfer_model = DecomposableAttentionChanged.from_params(vocab=trained_model.vocab,
                                                                  params=Params(transfer_model_params).duplicate())

        # TextFieldEmbedder parameters configured "freeze" so,
        # (1) transferred parameters should be same as trained parameters,
        for trained_parameter, transfer_parameter in zip(trained_model._text_field_embedder.parameters(),
                                                         transfer_model._text_field_embedder.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        # (2) transferred parameters' require grad should be Off.
        for parameter in transfer_model._text_field_embedder.parameters():
            assert not parameter.requires_grad

        # AttendFeedforward parameters configured "tune" so,
        # (1) transferred parameters should be same as trained parameters.
        for trained_parameter, transfer_parameter in zip(trained_model._attend_feedforward.parameters(),
                                                         transfer_model._attend_feedforward.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        # (2) transferred parameters' require grad should be On.
        for parameter in transfer_model._attend_feedforward.parameters():
            assert parameter.requires_grad

        # CompareFeedforward parameters configured "reinitialize" so,
        # (1) transferred parameters should not be same as trained parameters.
        for trained_parameter, transfer_parameter in zip(trained_model._compare_feedforward.parameters(),
                                                         transfer_model.scratch_compare_feedforward.parameters()):
            assert torch.all(trained_parameter == transfer_parameter)
        # (2) transferred parameters' require grad should be On.
        for parameter in transfer_model.scratch_compare_feedforward.parameters():
            assert parameter.requires_grad
