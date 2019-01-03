import json
from typing import Dict, Tuple, TYPE_CHECKING
import warnings

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import TokenIndexer, Token
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import remove_sentence_boundaries, get_text_field_mask, add_sentence_boundary_token_ids

# Importing at runtime results in a circular dependency.
if TYPE_CHECKING:
    from allennlp.models.bidirectional_lm import BidirectionalLanguageModel


@TokenEmbedder.register('bidirectional_lm_token_embedder')
class BidirectionalLanguageModelTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of representations from a bidirectional language model. This is done
    by computing a learned scalar average of the layers from the LM. Typically the LM's weights
    will be fixed, but they can be fine tuned by setting ``requires_grad``.

    .. deprecated:: 0.8
        ``BidirectionalLanguageModelTokenEmbedder`` was deprecated in version 0.8
        and will be removed in version 0.10 .

    Parameters
    ----------
    archive_file : ``str``, required
        An archive file, typically model.tar.gz, from a BidirectionalLanguageModel. The
        contextualizer used by the LM must satisfy two requirements:

        1. It must have a num_layers field.
        2. It must take a boolean return_all_layers parameter in its constructor.

        See BidirectionalLanguageModelTransformer for their definitions.

    dropout : ``float``, optional.
        The dropout value to be applied to the representations.
    bos_eos_tokens : ``Tuple[str, str]``, optional (default=``("<S>", "</S>")``)
        These will be indexed and placed around the indexed tokens. Necessary if the language model
        was trained with them, but they were injected external to an indexer.
    remove_bos_eos: ``bool``, optional (default: True)
        Typically the provided token indexes will be augmented with begin-sentence and end-sentence
        tokens. (Alternatively, you can pass bos_eos_tokens.) If this flag is True the
        corresponding embeddings will be removed from the return values.

        Warning: This only removes a single start and single end token!
    requires_grad : ``bool``, optional (default: False)
        If True, compute gradient of bidirectional language model parameters for fine tuning.
    """
    def __init__(self,
                 archive_file: str,
                 dropout: float = None,
                 bos_eos_tokens: Tuple[str, str] = ("<S>", "</S>"),
                 remove_bos_eos: bool = True,
                 requires_grad: bool = False) -> None:
                warnings.warn('BidirectionalLanguageModelTokenEmbedder is deprecated, '
                              'please use the ShuffledSentenceLanguageModelTokenEmbedder '
                              '(registered under "shuffled_sentence_lm_token_embedder").',
                      DeprecationWarning)
                super().__init__(archive_file=archive_file,
                                 dropout=dropout,
                                 bos_eos_tokens=bos_eos_tokens,
                                 remove_bos_eos=remove_bos_eos,
                                 requires_grad=requires_grad)
