from typing import Tuple

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders.language_model_token_embedder import (
    LanguageModelTokenEmbedder,
)


@TokenEmbedder.register("bidirectional_lm_token_embedder")
class BidirectionalLanguageModelTokenEmbedder(LanguageModelTokenEmbedder):
    """
    Compute a single layer of representations from a bidirectional language model. This is done
    by computing a learned scalar average of the layers from the LM. Typically the LM's weights
    will be fixed, but they can be fine tuned by setting `requires_grad`.

    # Parameters

    archive_file : `str`, required
        An archive file, typically model.tar.gz, from a BidirectionalLanguageModel. The
        contextualizer used by the LM must satisfy two requirements:

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
        super().__init__(
            archive_file=archive_file,
            dropout=dropout,
            bos_eos_tokens=bos_eos_tokens,
            remove_bos_eos=remove_bos_eos,
            requires_grad=requires_grad,
        )
