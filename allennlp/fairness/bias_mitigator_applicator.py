"""
A Model wrapper to mitigate biases in
contextual embeddings during finetuning
on a downstream task and test time.

Based on: Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020).
[On Measuring and Mitigating Biased Inferences of Word Embeddings]
(https://api.semanticscholar.org/CorpusID:201670701).
ArXiv, abs/1908.09369.
"""

from overrides import overrides

from allennlp.fairness.bias_mitigator_wrappers import BiasMitigatorWrapper

from allennlp.common.lazy import Lazy
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import find_embedding_layer


@Model.register("bias_mitigator_applicator")
class BiasMitigatorApplicator(Model):
    """
    Wrapper class to apply bias mitigation to any pretrained Model.

    # Parameters

    vocab : `Vocabulary`
        Vocabulary of base model.
    base_model : `Model`
        Base model for which to mitigate biases.
    bias_mitigator : `Lazy[BiasMitigatorWrapper]`
        Bias mitigator to apply to base model.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        base_model: Model,
        bias_mitigator: Lazy[BiasMitigatorWrapper],
        **kwargs
    ):
        super().__init__(vocab, **kwargs)

        self.base_model = base_model
        # want to keep bias mitigation hook during test time
        embedding_layer = find_embedding_layer(self.base_model)

        self.bias_mitigator = bias_mitigator.construct(embedding_layer=embedding_layer)
        embedding_layer.register_forward_hook(self.bias_mitigator)

        self.vocab = self.base_model.vocab
        self._regularizer = self.base_model._regularizer

    @overrides
    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.train(mode)
        # appropriately change requires_grad
        # in bias mitigator and bias direction
        # when train() and eval() are called
        self.bias_mitigator.train(mode)

    # Delegate Model function calls to base_model
    # Currently doing this manually because difficult to
    # dynamically forward __getattribute__ due to
    # behind-the-scenes usage of dunder attributes by torch.nn.Module
    # and both BiasMitigatorWrapper and base_model inheriting from Model
    # Assumes Model is relatively stable
    # TODO: adapt BiasMitigatorWrapper to changes in Model
    @overrides
    def forward(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)

    @overrides
    def forward_on_instance(self, *args, **kwargs):
        return self.base_model.forward_on_instance(*args, **kwargs)

    @overrides
    def forward_on_instances(self, *args, **kwargs):
        return self.base_model.forward_on_instances(*args, **kwargs)

    @overrides
    def get_regularization_penalty(self, *args, **kwargs):
        return self.base_model.get_regularization_penalty(*args, **kwargs)

    @overrides
    def get_parameters_for_histogram_logging(self, *args, **kwargs):
        return self.base_model.get_parameters_for_histogram_logging(*args, **kwargs)

    @overrides
    def get_parameters_for_histogram_tensorboard_logging(self, *args, **kwargs):
        return self.base_model.get_parameters_for_histogram_tensorboard_logging(*args, **kwargs)

    @overrides
    def make_output_human_readable(self, *args, **kwargs):
        return self.base_model.make_output_human_readable(*args, **kwargs)

    @overrides
    def get_metrics(self, *args, **kwargs):
        return self.base_model.get_metrics(*args, **kwargs)

    @overrides
    def _get_prediction_device(self, *args, **kwargs):
        return self.base_model._get_prediction_device(*args, **kwargs)

    @overrides
    def _maybe_warn_for_unseparable_batches(self, *args, **kwargs):
        return self.base_model._maybe_warn_for_unseparable_batches(*args, **kwargs)

    @overrides
    def extend_embedder_vocab(self, *args, **kwargs):
        return self.base_model.extend_embedder_vocab(*args, **kwargs)
