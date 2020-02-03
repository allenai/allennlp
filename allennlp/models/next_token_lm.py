from typing import Dict

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import LanguageModelHead, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import Perplexity


@Model.register("next_token_lm")
class NextTokenLM(Model):
    """
    The `NextTokenLM` embeds some input tokens, contextualizes them, then predicts the next word,
    computing a loss against known target.

    NOTE: This was developed for use in a demo, not for training.  You `definitely` don't want to
    train a language model using this code; it would be incredibly inefficient.  This `does`
    compute correct gradients of the loss, however, so you can use it for interesting visualization
    of the gradients of a pretrained model, and it appears to be fast enough to sample from, at
    least for one word at a time.  If you want to sample many tokens at a time, you'd want to
    re-use some intermediate computation, so you would either need to modify this code or use
    something else.

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
        self, tokens: TextFieldTensors, target_ids: TextFieldTensors = None
    ) -> Dict[str, torch.Tensor]:

        # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self._text_field_embedder(tokens)
        batch_size = embeddings.size(0)

        # Shape: (batch_size, num_tokens, encoding_dim)
        if self._contextualizer:
            mask = util.get_text_field_mask(embeddings)
            contextual_embeddings = self._contextualizer(embeddings, mask)
            final_embeddings = util.get_final_encoder_states(contextual_embeddings, mask)
        else:
            final_embeddings = embeddings[:, -1]

        target_logits = self._language_model_head(self._dropout(final_embeddings))

        vocab_size = target_logits.size(-1)
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        k = min(vocab_size, 5)  # min here largely because tests use small vocab
        top_probs, top_indices = probs.topk(k=k, dim=-1)

        output_dict = {"probabilities": top_probs, "top_indices": top_indices}

        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if target_ids is not None:
            targets = util.get_token_ids_from_text_field_tensors(target_ids).view(batch_size)
            target_logits = target_logits.view(batch_size, vocab_size)
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
                        for index in instance_indices
                    ]
                ]
            )
            output_dict["words"] = top_words
        tokens = []
        print(output_dict["token_ids"])
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
