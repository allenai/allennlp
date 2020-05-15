from typing import List
import torch

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.time_distributed import TimeDistributed


@TokenEmbedder.register("elmo_token_embedder")
class ElmoTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of ELMo representations.

    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...)

    Registered as a `TokenEmbedder` with name "elmo_token_embedder".

    # Parameters

    options_file : `str`, required.
        An ELMo JSON options file.
    weight_file : `str`, required.
        An ELMo hdf5 weight file.
    do_layer_norm : `bool`, optional.
        Should we apply layer normalization (passed to `ScalarMix`)?
    dropout : `float`, optional, (default = `0.5`).
        The dropout value to be applied to the ELMo representations.
    requires_grad : `bool`, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    projection_dim : `int`, optional
        If given, we will project the ELMo embedding down to this dimension.  We recommend that you
        try using ELMo with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particularly where there is very limited training data).
    vocab_to_cache : `List[str]`, optional.
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, the ElmoTokenEmbedder expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    scalar_mix_parameters : `List[int]`, optional, (default=`None`)
        If not `None`, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training. The mixing weights here should be the unnormalized (i.e., pre-softmax)
        weights. So, if you wanted to use only the 1st layer of a 2-layer ELMo,
        you can set this to [-9e10, 1, -9e10 ].
    """

    def __init__(
        self,
        options_file: str = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/"
        + "elmo_2x4096_512_2048cnn_2xhighway_options.json",
        weight_file: str = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/"
        + "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        requires_grad: bool = False,
        projection_dim: int = None,
        vocab_to_cache: List[str] = None,
        scalar_mix_parameters: List[float] = None,
    ) -> None:
        super().__init__()

        self._elmo = Elmo(
            options_file,
            weight_file,
            1,
            do_layer_norm=do_layer_norm,
            dropout=dropout,
            requires_grad=requires_grad,
            vocab_to_cache=vocab_to_cache,
            scalar_mix_parameters=scalar_mix_parameters,
        )
        if projection_dim:
            self._projection = torch.nn.Linear(self._elmo.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._elmo.get_output_dim()

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, elmo_tokens: torch.Tensor, word_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        # Parameters

        elmo_tokens : `torch.Tensor`
            Shape `(batch_size, timesteps, 50)` of character ids representing the current batch.
        word_inputs : `torch.Tensor`, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            `(batch_size, timesteps)`, which represent word ids which have been pre-cached.

        # Returns

        `torch.Tensor`
            The ELMo representations for the input sequence, shape
            `(batch_size, timesteps, embedding_dim)`
        """
        elmo_output = self._elmo(elmo_tokens, word_inputs)
        elmo_representations = elmo_output["elmo_representations"][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations
