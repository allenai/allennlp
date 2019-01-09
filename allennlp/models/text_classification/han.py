from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask


@Model.register("han")
class HierarchicalAttentionNetwork(Model):
    """
    This ``Model`` implements the Hierarchical Attention Network described in
    https://www.semanticscholar.org/paper/Hierarchical-Attention-Networks-for-Document-Yang-Yang/1967ad3ac8a598adc6929e9e6b9682734f789427
    by Yang et. al, 2016.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``TextField`` we get as input to the model.
    word_encoder : ``Seq2SeqEncoder``
        Used to encode words.
    sentence_encoder : ``Seq2SeqEncoder``
        Used to encode sentences.
    word_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention
        to calculate a fixed-length vector representation of each sentence
        from that sentence's sequence of word vectors
    sentence_attention : ``Seq2VecEncoder``
        Seq2Vec layer that (in original implementation) uses attention to
        calculate a fixed-length vector representation of each document from
        that document's sequence of sentence vectors
    classification_layer : ``FeedForward``
        This feedforward network computes the output logits.
    pre_word_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before word_attention encoder.
    pre_sentence_encoder_dropout : ``float``, optional (default=0.0)
        Dropout percentage to use before sentence_attention encoder.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 word_encoder: Seq2SeqEncoder,
                 sentence_encoder: Seq2SeqEncoder,
                 word_attention: Seq2VecEncoder,
                 sentence_attention: Seq2VecEncoder,
                 classification_layer: FeedForward,
                 word_dropout: float = 0.0,
                 sentence_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)
        self._pad_idx = self.vocab.get_token_index("@@PADDING@@")
        self._text_field_embedder = text_field_embedder
        self._word_attention = word_attention
        self._sentence_attention = sentence_attention
        self._word_dropout = torch.nn.Dropout(p=word_dropout)
        self._word_encoder = word_encoder
        self._sentence_dropout = torch.nn.Dropout(p=sentence_dropout)
        self._sentence_encoder = sentence_encoder
        self._classification_layer = classification_layer
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``. These tokens should be segmented into sentences.
            3-d tensor: all_docs x max_num_sents_in_any_doc x max_num_tokens_in_any_doc_sent
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        tokens_ = tokens['tokens']
        batch_size = tokens_.size(0)
        max_num_sents = tokens_.size(1)
        sentence_level_mask = ((tokens_ == self._pad_idx).all(dim=2) == 0).float()

        embedded_words = self._text_field_embedder(tokens)
        batch_size, _, _, _ = embedded_words.size()
        embedded_words = embedded_words.view(batch_size * max_num_sents, embedded_words.size(2), -1)
        tokens_ = tokens_.view(batch_size * max_num_sents, -1)

        # we encode words with a seq2seq encoder
        # then apply attention to get sentence-level representation
        mask = get_text_field_mask({"tokens": tokens_}).float()
        embedded_words = self._word_dropout(embedded_words)
        encoded_words = self._word_encoder(embedded_words, mask)
        sentence_repr = self._word_attention(encoded_words, mask)
        sentence_repr = sentence_repr.view(batch_size, max_num_sents, -1)

        # we encode sentences with a seq2seq encoder
        # then apply attention to get document-level representation
        sentence_repr = self._sentence_dropout(sentence_repr)
        encoded_sents = self._sentence_encoder(sentence_repr, sentence_level_mask)
        document_repr = self._sentence_attention(encoded_sents, sentence_level_mask)

        label_logits = self._classification_layer(document_repr.view(batch_size, -1))
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
