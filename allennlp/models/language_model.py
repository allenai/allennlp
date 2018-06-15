from typing import Optional, Dict

from overrides import overrides
import torch
import torch.nn as nn

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, SoftmaxWithNLL
from allennlp.common.checks import check_dimensions_match
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from ipdb import set_trace

from allennlp.training.metrics.perplexity import Perplexity

@Model.register("word-lm")
class WordLM(Model):
    """
    This ``WordLM`` is a word level language model.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``LMRNN``, required
        The Recurrent Neural Networks for language modeling.
    softmax : ``AdaptiveSoftmax``
        The adaptive softmax for predicting next words
    proj: bool, optional (default=``True``)
        whether add a linear projection before the softmax layer
    relu: bool, optional (default=``True``)
        whether add a ReLU non-linear transformation before the softmax layer
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 softmax: SoftmaxWithNLL,
                 proj: bool = False,
                 relu: bool = False,
                 dropout: float = 0.0,
                 batch_first: bool=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)

        self.encoder = encoder
        self.batch_first = batch_first
        self.text_field_embedder = text_field_embedder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
            dropout_list = [self.dropout]
        else:
            self.dropout = None
            dropout_list = []

        if proj:
            self.proj = nn.Linear(self.encoder.get_output_dim(), softmax.get_input_dim())
            proj_list = [self.proj] 
            if relu:
                proj_list += [nn.ReLU()]
            proj_list = proj_list + dropout_list
        else:
            self.proj = None
            proj_list = []

        after_encoder_list = dropout_list + proj_list
        if len(after_encoder_list) > 0:
            self.after_encoder = nn.Sequential(*after_encoder_list)
        else:
            self.after_encoder = None

        self.softmax = softmax
        if not self.softmax.adaptive and self.softmax.head.weight.size() == self.text_field_embedder.token_embedder_tokens.weight.size():
            self.softmax.head.weight = self.text_field_embedder.token_embedder_tokens.weight

        self.softmax_in = softmax.get_input_dim()

        self.metrics = {"ppl": Perplexity()}

        check_dimensions_match(text_field_embedder.get_output_dim(), 
            encoder.get_input_dim(), 
            "text field embedder output dim",
            "encoder input dim")
        if not proj:
            check_dimensions_match(encoder.get_output_dim(), 
                self.softmax_in,
                "encoder output dim",
                "softmax input dim")

        initializer(self)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        
    @overrides
    def train(self, mode=True):
        self.encoder.reset_states()
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    @overrides
    def eval(self):
        return self.train(False)

    @overrides
    def forward(self,  # type: ignore
                input_tokens: Dict[str, torch.LongTensor],
                output_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        output_tokens : Dict[str, torch.LongTensor], optional
            The expected prediction of the language model. If set to be None, the output would contains
            ```emb_encoder_out```, which is the output of the LMRNNs. Otherwise, the output would be the loss
            alone.

        Returns
        -------
        An output dictionary consisting of:
        emb_encoder_out : torch.FloatTensor, optional
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """

        embedded_text_input = self.text_field_embedder(input_tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        # set_trace()
        emb_encoder_out = self.encoder(embedded_text_input, 
            embedded_text_input.new_ones((embedded_text_input.size(0), embedded_text_input.size(1))))

        # set_trace()
        if self.after_encoder:
            emb_encoder_out = self.after_encoder(emb_encoder_out)

        if output_tokens is None:
            output_dict = {"emb_encoder_out": emb_encoder_out}
        
        else:
            nll = self.softmax(emb_encoder_out, output_tokens["tokens"])

            for metric in self.metrics.values():
                metric(nll.data.item())

            output_dict = {"loss" : nll}

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'WordLM':

        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)

        encoder_params = params.pop("encoder")
        encoder = Seq2SeqEncoder.from_params(encoder_params)

        proj = params.pop("proj", False)
        relu = params.pop("relu", False)
        dropout = params.pop("dropout", None)

        softmax_params = params.pop('softmax')
        softmax = SoftmaxWithNLL.from_params(vocab, softmax_params)
        
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                 text_field_embedder=text_field_embedder,
                 encoder=encoder,
                 softmax=softmax,
                 proj=proj,
                 relu=relu,
                 dropout=dropout,
                 initializer=initializer,
                 regularizer=regularizer)