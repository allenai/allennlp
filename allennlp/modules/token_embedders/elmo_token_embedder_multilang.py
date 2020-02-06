from typing import List, Dict
import torch

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.time_distributed import TimeDistributed


@TokenEmbedder.register("elmo_token_embedder_multilang")
class ElmoTokenEmbedderMultiLang(TokenEmbedder):
    """
    A multilingual ELMo embedder - extending ElmoTokenEmbedder for multiple languages.
    Each language has different weights for the ELMo model and an alignment matrix.

    # Parameters

    options_files : `Dict[str, str]`, required.
        A dictionary of language identifier to an ELMo JSON options file.
    weight_files : `Dict[str, str]`, required.
        A dictionary of language identifier to an ELMo hdf5 weight file.
    do_layer_norm : `bool`, optional.
        Should we apply layer normalization (passed to `ScalarMix`)?
    dropout : `float`, optional.
        The dropout value to be applied to the ELMo representations.
    requires_grad : `bool`, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    projection_dim : `int`, optional
        If given, we will project the ELMo embedding down to this dimension.  We recommend that you
        try using ELMo with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particulary where there is very limited training data).
    vocab_to_cache : `List[str]`, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, the ElmoTokenEmbedder expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    scalar_mix_parameters : `List[int]`, optional, (default=None).
        If not `None`, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    aligning_files : `Dict[str, str]`, optional, (default={}).
        A dictionary of language identifier to a pth file with an alignment matrix.
    """

    def __init__(
        self,
        options_files: Dict[str, str],
        weight_files: Dict[str, str],
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        requires_grad: bool = False,
        projection_dim: int = None,
        vocab_to_cache: List[str] = None,
        scalar_mix_parameters: List[float] = None,
        aligning_files: Dict[str, str] = None,
    ) -> None:
        super().__init__()

        if options_files.keys() != weight_files.keys():
            raise ConfigurationError("Keys for Elmo's options files and weights files don't match")

        aligning_files = aligning_files or {}
        output_dim = None
        for lang in weight_files.keys():
            name = "elmo_%s" % lang
            elmo = Elmo(
                options_files[lang],
                weight_files[lang],
                num_output_representations=1,
                do_layer_norm=do_layer_norm,
                dropout=dropout,
                requires_grad=requires_grad,
                vocab_to_cache=vocab_to_cache,
                scalar_mix_parameters=scalar_mix_parameters,
            )
            self.add_module(name, elmo)

            output_dim_tmp = elmo.get_output_dim()
            if output_dim is not None:
                # Verify that all ELMo embedders have the same output dimension.
                check_dimensions_match(
                    output_dim_tmp, output_dim, "%s output dim" % name, "elmo output dim"
                )

            output_dim = output_dim_tmp

        self.output_dim = output_dim

        if projection_dim:
            self._projection = torch.nn.Linear(output_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None

        for lang in weight_files.keys():
            name = "aligning_%s" % lang
            aligning_matrix = torch.eye(output_dim)
            if lang in aligning_files and aligning_files[lang] != "":
                aligninig_path = cached_path(aligning_files[lang])
                aligning_matrix = torch.FloatTensor(torch.load(aligninig_path))

            aligning = torch.nn.Linear(output_dim, output_dim, bias=False)
            aligning.weight = torch.nn.Parameter(aligning_matrix, requires_grad=False)
            self.add_module(name, aligning)

    def get_output_dim(self):
        return self.output_dim

    def forward(
        self, tokens: torch.Tensor, lang: str, word_inputs: torch.Tensor = None
    ) -> torch.Tensor:
        """
        # Parameters

        tokens : `torch.Tensor`
            Shape `(batch_size, timesteps, 50)` of character ids representing the current batch.
        lang : `str`, , required.
            The language of the ELMo embedder to use.
        word_inputs : `torch.Tensor`, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            `(batch_size, timesteps)`, which represent word ids which have been pre-cached.

        # Returns

        The ELMo representations for the given language for the input sequence, shape
        `(batch_size, timesteps, embedding_dim)`
        """
        elmo = getattr(self, "elmo_{}".format(lang))
        elmo_output = elmo(tokens, word_inputs)
        elmo_representations = elmo_output["elmo_representations"][0]
        aligning = getattr(self, "aligning_{}".format(lang))
        elmo_representations = aligning(elmo_representations)
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations
