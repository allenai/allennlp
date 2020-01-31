from typing import Dict

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import LanguageModelHead, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import Perplexity


@Model.register("masked_language_model")
class MaskedLanguageModel(Model):
    """
    The `MaskedLanguageModel` embeds some input tokens (including some which are masked),
    contextualizes them, then predicts targets for the masked tokens, computing a loss against
    known targets.

    NOTE: This was developed for use in a demo, not for training.  It's possible that it will still
    work for training a masked LM, but it is very likely that some other code would be much more
    efficient for that.  This `does` compute correct gradients of the loss, because we use that in
    our demo, so in principle it should be able to train a model, we just don't necessarily endorse
    that use.

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the indexed tokens we get in `forward`.
    language_model_head : `LanguageModelHead`
        The `torch.nn.Module` that goes from the hidden states output by the contextualizer to
        logits over some output vocabulary.
    contextualizer : `Seq2SeqEncoder`, optional (default=None)
        Used to "contextualize" the embeddings.  This is optional because the contextualization
        might actually be done in the text field embedder.
    target_namespace : `str`, optional (default='bert')
        Namespace to use to convert predicted token ids to strings in `Model.decode`.
    dropout : `float`, optional (default=0.0)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        language_model_head: LanguageModelHead,
        contextualizer: Seq2SeqEncoder = None,
        target_namespace: str = "bert",
        dropout: float = 0.0,
        initializer: InitializerApplicator = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._contextualizer = contextualizer
        if contextualizer:
            check_dimensions_match(
                text_field_embedder.get_output_dim(),
                contextualizer.get_input_dim(),
                "text field embedder output",
                "contextualizer input",
            )
        self._language_model_head = language_model_head
        self._target_namespace = target_namespace
        self._perplexity = Perplexity()
        self._dropout = torch.nn.Dropout(dropout)

        if initializer is not None:
            initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        mask_positions: torch.LongTensor,
        target_ids: TextFieldTensors = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        tokens : `TextFieldTensors`
            The output of `TextField.as_tensor()` for a batch of sentences.
        mask_positions : `torch.LongTensor`
            The positions in `tokens` that correspond to [MASK] tokens that we should try to fill
            in.  Shape should be (batch_size, num_masks).
        target_ids : `TextFieldTensors`
            This is a list of token ids that correspond to the mask positions we're trying to fill.
            It is the output of a `TextField`, purely for convenience, so we can handle wordpiece
            tokenizers and such without having to do crazy things in the dataset reader.  We assume
            that there is exactly one entry in the dictionary, and that it has a shape identical to
            `mask_positions` - one target token per mask position.
        """

        targets = None
        if target_ids is not None:
            # A bit of a hack to get the right targets out of the TextField output...
            if len(target_ids) != 1:
                targets = target_ids["bert"]["token_ids"]
            else:
                targets = list(target_ids.values())[0]["tokens"]
        mask_positions = mask_positions.squeeze(-1)
        batch_size, num_masks = mask_positions.size()
        if targets is not None and targets.size() != mask_positions.size():
            raise ValueError(
                f"Number of targets ({targets.size()}) and number of masks "
                f"({mask_positions.size()}) are not equal"
            )

        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self._text_field_embedder(tokens)

        # Shape: (batch_size, num_tokens, encoding_dim)
        if self._contextualizer:
            mask = util.get_text_field_mask(embeddings)
            contextual_embeddings = self._contextualizer(embeddings, mask)
        else:
            contextual_embeddings = embeddings

        # Does advanced indexing to get the embeddings of just the mask positions, which is what
        # we're trying to predict.
        batch_index = torch.arange(0, batch_size).long().unsqueeze(1)
        mask_embeddings = contextual_embeddings[batch_index, mask_positions]

        target_logits = self._language_model_head(self._dropout(mask_embeddings))

        vocab_size = target_logits.size(-1)
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        k = min(vocab_size, 5)  # min here largely because tests use small vocab
        top_probs, top_indices = probs.topk(k=k, dim=-1)

        output_dict = {"probabilities": top_probs, "top_indices": top_indices}

        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if targets is not None:
            target_logits = target_logits.view(batch_size * num_masks, vocab_size)
            targets = targets.view(batch_size * num_masks)
            loss = torch.nn.functional.cross_entropy(target_logits, targets)
            self._perplexity(loss)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        top_words = []
        for instance_indices in output_dict["top_indices"]:
            top_words.append(
                [
                    [
                        self.vocab.get_token_from_index(
                            index.item(), namespace=self._target_namespace
                        )
                        for index in mask_positions
                    ]
                    for mask_positions in instance_indices
                ]
            )
        output_dict["words"] = top_words
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        token_id.item(), namespace=self._target_namespace
                    )
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens

        return output_dict
