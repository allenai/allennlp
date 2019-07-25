from typing import Dict, List, Tuple, Union

from overrides import overrides
import torch

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import Perplexity


@Model.register('next_token_lm')
class NextTokenLM(Model):    
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 contextualizer: Seq2SeqEncoder = None,
                 target_namespace: str = 'bert',
                 dropout: float = 0.0,
                 initializer: InitializerApplicator = None) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._contextualizer = contextualizer
        self._target_namespace = target_namespace
        if contextualizer:
            softmax_dim = contextualizer.get_output_dim()
            check_dimensions_match(text_field_embedder.get_output_dim(), contextualizer.get_input_dim(),
                                   "text field embedder output", "contextualizer input")
        else:
            softmax_dim = text_field_embedder.get_output_dim()
        self._language_model_head = torch.nn.Linear(softmax_dim,
                                                    vocab.get_vocab_size(target_namespace))
        self._perplexity = Perplexity()
        self._dropout = torch.nn.Dropout(dropout)

        if initializer is not None:
            initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],                
                target_ids: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:        
        # pylint: disable=arguments-differ        
        batch_size = tokens['tokens'].size()[0]
        
         # Shape: (batch_size, num_tokens, embedding_dim)
        embeddings = self._text_field_embedder(tokens)

         # Shape: (batch_size, num_tokens, encoding_dim)
        if self._contextualizer:
            mask = util.get_text_field_mask(embeddings)
            contextual_embeddings = self._contextualizer(embeddings, mask)
        else:
            contextual_embeddings = embeddings

        batch_index = torch.arange(0, batch_size).long().unsqueeze(1)        
        final_embeddings = contextual_embeddings[batch_index, -1]

        target_logits = self._language_model_head(self._dropout(final_embeddings))

        vocab_size = target_logits.size(-1)
        probs = torch.nn.functional.softmax(target_logits, dim=-1)
        k = min(vocab_size, 20)  # min here largely because tests use small vocab
        top_probs, top_indices = probs.topk(k=k, dim=-1)

        output_dict = {"top_probs": top_probs, "top_indices": top_indices}

        if target_ids is not None:
            target_ids = list(target_ids.values())[0]        
            target_logits = target_logits.view(batch_size, vocab_size)
            target_ids = target_ids.view(batch_size)
            loss = torch.nn.functional.cross_entropy(target_logits, target_ids)
            self._perplexity(loss)
            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual target_ids.
        ``output_dict["target_ids"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["target_ids"] = [
                [self.vocab.get_token_from_index(target_id, namespace=self._target_namespace)
                 for target_id in target_ids]
                for target_ids in output_dict["target_ids"]
        ]

        return output_dict
