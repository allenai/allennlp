import json
from typing import Dict, Tuple, TYPE_CHECKING

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer, Token
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import (
    remove_sentence_boundaries,
    get_text_field_mask,
    add_sentence_boundary_token_ids,
)

# Importing at runtime results in a circular dependency.
if TYPE_CHECKING:
    from allennlp.models.language_model import LanguageModel


@TokenEmbedder.register("language_model_token_embedder")
class LanguageModelTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of representations from a (optionally bidirectional)
    language model. This is done by computing a learned scalar
    average of the layers from the LM. Typically the LM's weights
    will be fixed, but they can be fine tuned by setting `requires_grad`.

    # Parameters

    archive_file : `str`, required
        An archive file, typically model.tar.gz, from a LanguageModel.
        The contextualizer used by the LM must satisfy two requirements:

        1. It must have a num_layers field.
        2. It must take a boolean return_all_layers parameter in its constructor.

        See BidirectionalLanguageModelTransformer for their definitions.

    dropout : `float`, optional.
        The dropout value to be applied to the representations.
    bos_eos_tokens : `Tuple[str, str]`, optional (default=`("<S>", "</S>")`)
        These will be indexed and placed around the indexed tokens. Necessary if the language model
        was trained with them, but they were injected external to an indexer.
    remove_bos_eos : `bool`, optional (default: True)
        Typically the provided token indexes will be augmented with begin-sentence and end-sentence
        tokens. (Alternatively, you can pass bos_eos_tokens.) If this flag is True the
        corresponding embeddings will be removed from the return values.

        Warning: This only removes a single start and single end token!
    requires_grad : `bool`, optional (default: False)
        If True, compute gradient of bidirectional language model parameters for fine tuning.
    """

    def __init__(
        self,
        archive_file: str,
        dropout: float = None,
        bos_eos_tokens: Tuple[str, str] = ("<S>", "</S>"),
        remove_bos_eos: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        overrides = {"model": {"contextualizer": {"return_all_layers": True}}}

        # Import here to avoid circular dependency.
        from allennlp.models.archival import load_archive

        # Load LM and the associated config.
        archive = load_archive(archive_file, overrides=json.dumps(overrides))
        self._lm: LanguageModel = archive.model
        self._lm.delete_softmax()
        config = archive.config
        dict_config = config.as_dict(quiet=True)

        # Extract the name of the tokens that the LM was trained on.
        text_field_embedder = dict_config["model"]["text_field_embedder"]
        token_names = list(text_field_embedder["token_embedders"].keys())
        if len(token_names) != 1:
            # We don't currently support embedding with language models trained with multiple
            # embedded indices.
            #
            # Note: We only care about embedded indices. This does not include "tokens" which
            # is just used to compute the loss in LanguageModel.
            raise ConfigurationError(f"LM from {archive_file} trained with multiple embedders!")
        self._token_name = token_names[0]

        # TODO(brendanr): Find a way to remove this hack. The issue fundamentally is that the
        # BasicTextFieldEmbedder concatenates multiple embedded representations. When a
        # downstream model uses both, tokens and token characters, say, and only adds bos/eos
        # tokens to the token characters, the dimensions don't match. See:
        # https://github.com/allenai/allennlp/blob/eff25a3085aa9976a7650d30d8961c3626ddc411/allennlp/modules/text_field_embedders/basic_text_field_embedder.py#L109
        #
        # For the equivalent hack in the ELMo embedder see:
        # https://github.com/allenai/allennlp/blob/eff25a3085aa9976a7650d30d8961c3626ddc411/allennlp/modules/elmo.py#L590
        if bos_eos_tokens:
            dataset_reader_config = config.get("dataset_reader")
            if dataset_reader_config.get("type") == "multiprocess":
                dataset_reader_config = dataset_reader_config.get("base_reader")
            token_indexer_config = dataset_reader_config.get("token_indexers").get(self._token_name)
            token_indexer: TokenIndexer = TokenIndexer.from_params(token_indexer_config)
            token_list = [Token(token) for token in bos_eos_tokens]
            # TODO(brendanr): Obtain these indices from the vocab once the
            # ELMoTokenCharactersIndexer adds the mappings.
            bos_eos_indices = token_indexer.tokens_to_indices(token_list, self._lm.vocab)["tokens"]
            self._bos_indices = torch.LongTensor(bos_eos_indices[0])
            self._eos_indices = torch.LongTensor(bos_eos_indices[1])
        else:
            self._bos_indices = None
            self._eos_indices = None

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        self._remove_bos_eos = remove_bos_eos
        num_layers = self._lm.num_layers()
        # TODO(brendanr): Consider passing our LM as a custom module to `Elmo` instead.
        # See https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L76
        self._scalar_mix = ScalarMix(mixture_size=num_layers, do_layer_norm=False, trainable=True)

        character_dim = self._lm._text_field_embedder.get_output_dim()
        contextual_dim = self._lm._contextualizer.get_output_dim()

        if contextual_dim % character_dim != 0:
            raise ConfigurationError(
                "The output dimensions for the text_field_embedder "
                + f"({character_dim}) and the contextualizer ({contextual_dim})"
                + f" from the language model loaded from {archive_file} are "
                + "not compatible. Please check the config used to train that "
                + "model and ensure that the output dimension of the "
                + "text_field_embedder divides the output dimension of the "
                + "contextualizer."
            )
        self._character_embedding_duplication_count = contextual_dim // character_dim

        for param in self._lm.parameters():
            param.requires_grad = requires_grad

    def get_output_dim(self) -> int:
        return self._lm._contextualizer.get_output_dim()

    def forward(
        self,  # type: ignore
        tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        tokens : `torch.Tensor`
            Shape `(batch_size, timesteps, ...)` of token ids representing the current batch.
            These must have been produced using the same indexer the LM was trained on.

        # Returns

        The bidirectional language model representations for the input sequence, shape
        `(batch_size, timesteps, embedding_dim)`
        """

        if self._bos_indices is not None:
            num_wrapping_dims = max(tokens.dim() - 2, 0)
            mask = get_text_field_mask({"": {"": tokens}}, num_wrapping_dims=num_wrapping_dims)
            tokens, mask = add_sentence_boundary_token_ids(
                tokens, mask, self._bos_indices, self._eos_indices
            )

        source = {self._token_name: {"token_characters": tokens}}
        result_dict = self._lm(source)

        # shape (batch_size, timesteps, embedding_size)
        noncontextual_token_embeddings = result_dict["noncontextual_token_embeddings"]
        contextual_embeddings = result_dict["lm_embeddings"]

        # Typically the non-contextual embeddings are smaller than the contextualized embeddings.
        # Since we're averaging all the layers we need to make their dimensions match. Simply
        # repeating the non-contextual embeddings is a crude, but effective, way to do this.
        duplicated_character_embeddings = torch.cat(
            [noncontextual_token_embeddings] * self._character_embedding_duplication_count, -1
        )
        averaged_embeddings = self._scalar_mix(
            [duplicated_character_embeddings] + contextual_embeddings
        )

        # Add dropout
        averaged_embeddings = self._dropout(averaged_embeddings)
        if self._remove_bos_eos:
            averaged_embeddings, _ = remove_sentence_boundaries(
                averaged_embeddings, result_dict["mask"]
            )

        return averaged_embeddings
