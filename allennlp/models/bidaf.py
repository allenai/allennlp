from typing import Dict

import torch

from allennlp.common import Params, constants
from allennlp.common.tensor import get_text_field_mask, masked_softmax, last_dim_softmax, weighted_sum
from allennlp.data import Vocabulary
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.training import Model


class BidirectionalAttentionFlow(Model):
    """
    This class implements Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_
    for answering reading comprehension questions (ICLR 2017).

    The basic layout is pretty simple: encode words as a combination of word embeddings and a
    character-level encoder, pass the word representations through a bi-LSTM/GRU, use a matrix of
    attentions to put question information into the passage word representations (this is the only
    part that is at all non-standard), pass this through another few layers of bi-LSTMs/GRUs, and
    do a softmax over span start and span end.

    To instantiate this model with parameters matching those in the original paper, simply use
    ``BidirectionalAttentionFlow.from_params(vocab, Params({}))``.  This will construct all of the
    various dependencies needed for the constructor for you.

    Parameters
    ----------
    vocab : ``Vocabulary``
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    num_highway_layers : ``int``
        The number of highway layers to use in between embedding the input and passing it through
        the phrase layer.
    phrase_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and doing the bidirectional attention.
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between the bidirectional
        attention and predicting span start and end.
    span_end_encoder : ``Seq2SeqEncoder``
        The encoder that we will use to incorporate span start predictions into the passage state
        before predicting span end.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 attention_similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder) -> None:
        super(BidirectionalAttentionFlow, self).__init__()

        self._vocab = vocab
        self._text_field_embedder = text_field_embedder
        self._highway_layer = TimeDistributed(Highway(text_field_embedder.get_output_dim(),
                                                      num_highway_layers))
        self._phrase_layer = phrase_layer
        self._matrix_attention = MatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self._span_end_encoder = span_end_encoder

        encoding_dim = phrase_layer.get_output_dim()
        modeling_dim = modeling_layer.get_output_dim()
        span_start_input_dim = encoding_dim * 4 + modeling_dim
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(span_start_input_dim, 1))

        span_end_encoding_dim = span_end_encoder.get_output_dim()
        span_end_input_dim = encoding_dim * 4 + span_end_encoding_dim
        self._span_end_predictor = TimeDistributed(torch.nn.Linear(span_end_input_dim, 1))

        # TODO(Mark): support masking once utility functions are merged.
        self._loss = torch.nn.CrossEntropyLoss()

        # TODO(mattg): figure out default initialization here

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``
        span_start : torch.IntTensor, optional (default = None)
            From an ``IndexField``
        span_end : torch.IntTensor, optional (default = None)
            From an ``IndexField``

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span start position.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalised log
            probabilities of the span end position (exclusive).
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = get_text_field_mask(question).float()
        passage_mask = get_text_field_mask(passage).float()

        encoded_question = self._phrase_layer(embedded_question)
        encoded_passage = self._phrase_layer(embedded_passage)
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = weighted_sum(encoded_question, passage_question_attention)

        # TODO(mattg): this needs to mask things before doing this max.
        # Shape: (batch_size, passage_length)
        question_passage_similarity = passage_question_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = weighted_sum(encoded_passage, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_question_passage_vector = question_passage_vector.unsqueeze(1).expand(batch_size,
                                                                                    passage_length,
                                                                                    encoding_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_passage = torch.cat([encoded_passage,
                                          passage_question_vectors,
                                          encoded_passage * passage_question_vectors,
                                          encoded_passage * tiled_question_passage_vector],
                                         dim=-1)

        modeled_passage = self._modeling_layer(final_merged_passage)
        modeling_dim = modeled_passage.size(-1)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = torch.cat([final_merged_passage, modeled_passage], dim=-1)
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = weighted_sum(modeled_passage, span_start_probs)
        # Shape: (batch_size, passage_length, modeling_dim)
        tiled_start_representation = span_start_representation.unsqueeze(1).expand(batch_size,
                                                                                   passage_length,
                                                                                   modeling_dim)

        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim * 3)
        span_end_representation = torch.cat([final_merged_passage,
                                             modeled_passage,
                                             tiled_start_representation,
                                             modeled_passage * tiled_start_representation],
                                            dim=-1)
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_span_end = self._span_end_encoder(span_end_representation)
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = torch.cat([final_merged_passage, encoded_span_end], dim=-1)
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_end_probs = masked_softmax(span_end_logits, passage_mask)

        output_dict = {"span_start_logits": span_start_logits,
                       "span_start_probs": span_start_probs,
                       "span_end_logits": span_end_logits,
                       "span_end_probs": span_end_probs}
        if span_start:
            # Negative log likelihood criterion takes integer labels, not one hot.
            if span_start.dim() == 2:
                _, span_start = span_start.max(-1)
            loss = self._loss(span_start_logits, span_start.view(-1))
            if span_end.dim() == 2:
                _, span_end = span_end.max(-1)
            loss += self._loss(span_end_logits, span_end.view(-1))
            output_dict["loss"] = loss

        return output_dict

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        """
        With an empty ``params`` argument, this will instantiate a BiDAF model with the same
        configuration as published in the original BiDAF paper, as long as you've set
        ``allennlp.common.constants.GLOVE_PATH`` to the location of your gzipped 100-dimensional
        glove vectors.

        If you want to change parameters, the keys in the ``params`` object must match the
        constructor arguments above.
        """
        default_embedder_params = {
                'tokens': {
                        'type': 'embedding',
                        'pretrained_file': constants.GLOVE_PATH,
                        'trainable': False
                        },
                'token_characters': {
                        'type': 'character_encoding',
                        'embedding': {
                                'embedding_dim': 8
                                },
                        'encoder': {
                                'type': 'cnn',
                                'embedding_dim': 8,
                                'num_filters': 100,
                                'ngram_filter_sizes': [5]
                                }
                        }
                }
        embedder_params = params.pop("text_field_embedder", default_embedder_params)
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop("num_highway_layers", 2)
        default_phrase_layer_params = {
                'type': 'lstm',
                'bidirectional': True,
                'input_size': 200,
                'hidden_size': 100,
                'num_layers': 1,
                }
        phrase_layer_params = params.pop("phrase_layer", default_phrase_layer_params)
        phrase_layer = Seq2SeqEncoder.from_params(phrase_layer_params)
        default_similarity_function_params = {
                'type': 'linear',
                'combination': 'x,y,x*y',
                'tensor_1_dim': 200,
                'tensor_2_dim': 200
                }
        similarity_function_params = params.pop("similarity_function", default_similarity_function_params)
        similarity_function = SimilarityFunction.from_params(similarity_function_params)
        default_modeling_layer_params = {
                'type': 'lstm',
                'bidirectional': True,
                'input_size': 800,
                'hidden_size': 100,
                'num_layers': 2,
                }
        modeling_layer_params = params.pop("modeling_layer", default_modeling_layer_params)
        modeling_layer = Seq2SeqEncoder.from_params(modeling_layer_params)
        default_span_end_encoder_params = {
                'type': 'lstm',
                'bidirectional': True,
                'input_size': 1400,
                'hidden_size': 100,
                'num_layers': 2,
                }
        span_end_encoder_params = params.pop("span_end_encoder", default_span_end_encoder_params)
        span_end_encoder = Seq2SeqEncoder.from_params(span_end_encoder_params)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   span_end_encoder=span_end_encoder)
