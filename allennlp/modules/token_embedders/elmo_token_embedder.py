import torch

from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.data import Vocabulary


@TokenEmbedder.register("elmo_token_embedder")
class ElmoTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of ELMo representations.

    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...)

    Parameters
    ----------
    options_file : ``str``, required.
        An ELMo JSON options file.
    weight_file : ``str``, required.
        An ELMo hdf5 weight file.
    do_layer_norm : ``bool``, optional.
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional.
        The dropout value to be applied to the ELMo representations.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 requires_grad: bool = False) -> None:
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file,
                          weight_file,
                          1,
                          do_layer_norm=do_layer_norm,
                          dropout=dropout,
                          requires_grad=requires_grad)

    def get_output_dim(self):
        # pylint: disable=protected-access
        return 2 * self._elmo._elmo_lstm._token_embedder.get_output_dim()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.

        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs)
        return elmo_output['elmo_representations'][0]

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ElmoTokenEmbedder':
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('weight_file')
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        requires_grad = params.pop('requires_grad', False)
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        dropout = params.pop_float("dropout", 0.5)
        params.assert_empty(cls.__name__)
        return cls(options_file, weight_file, do_layer_norm, dropout, requires_grad=requires_grad)
