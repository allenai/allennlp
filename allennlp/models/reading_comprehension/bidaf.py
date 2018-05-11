import logging
import numpy as np

from typing import Any, Dict, List, Optional

import torch
from torch.autograd import Variable
from torch.nn.functional import nll_loss, binary_cross_entropy_with_logits, binary_cross_entropy, cross_entropy

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy, MultiSquadEmAndF1

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def to_variable(obj):
    var_obj = Variable(obj)
    if torch.cuda.is_available():
        var_obj = var_obj.cuda()

    return var_obj

def flatten_answer(passage, passage_mask):
    return torch.masked_select(passage, passage_mask.byte())


@Model.register("bidaf")
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

    Parameters
    ----------
    vocab : ``Vocabulary``
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
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 attention_similarity_function: SimilarityFunction,
                 modeling_layer: Seq2SeqEncoder,
                 span_end_encoder: Seq2SeqEncoder,
                 dropout: float = 0.2,
                 mask_lstms: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BidirectionalAttentionFlow, self).__init__(vocab, regularizer)

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

        # Bidaf has lots of layer dimensions which need to match up - these aren't necessarily
        # obvious from the configuration files, so we check here.
        check_dimensions_match(modeling_layer.get_input_dim(), 4 * encoding_dim,
                               "modeling layer input dim", "4 * encoding dim")
        check_dimensions_match(text_field_embedder.get_output_dim(), phrase_layer.get_input_dim(),
                               "text field embedder output dim", "phrase layer input dim")
        check_dimensions_match(span_end_encoder.get_input_dim(), 4 * encoding_dim + 3 * modeling_dim,
                               "span end encoder input dim", "4 * encoding dim + 3 * modeling dim")

        self._span_start_accuracy = CategoricalAccuracy()
        self._span_end_accuracy = CategoricalAccuracy()
        self._span_accuracy = BooleanAccuracy()
        self._squad_metrics = MultiSquadEmAndF1()
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms
        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        span_start : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            beginning position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        span_end : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            ending position of the answer with the passage.  This is an `inclusive` token index.
            If this is given, we will compute a loss that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.

        Returns
        -------
        An output dictionary consisting of:
        span_start_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        span_start_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        span_end_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span end position (inclusive).
        span_end_probs : torch.FloatTensor
            The result of ``softmax(span_end_logits)``.
        best_span : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        best_span_str : List[str]
            If sufficient metadata was provided for the instances in the batch, we also return the
            string from the original passage that the model thinks is the best answer to the
            question.
        """
        #import pdb; pdb.set_trace()
        embedded_question = self._highway_layer(self._text_field_embedder(question))
        embedded_passage = self._highway_layer(self._text_field_embedder(passage))
        batch_size = embedded_question.size(0)
        passage_length = embedded_passage.size(1)
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        question_lstm_mask = question_mask if self._mask_lstms else None
        passage_lstm_mask = passage_mask if self._mask_lstms else None

        encoded_question = self._dropout(self._phrase_layer(embedded_question, question_lstm_mask))
        encoded_passage = self._dropout(self._phrase_layer(embedded_passage, passage_lstm_mask))
        encoding_dim = encoded_question.size(-1)

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = util.last_dim_softmax(passage_question_similarity, question_mask)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = util.replace_masked_values(passage_question_similarity,
                                                       question_mask.unsqueeze(1),
                                                       -1e7)
        # Shape: (batch_size, passage_length)
        question_passage_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        question_passage_attention = util.masked_softmax(question_passage_similarity, passage_mask)
        # Shape: (batch_size, encoding_dim)
        question_passage_vector = util.weighted_sum(encoded_passage, question_passage_attention)
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

        modeled_passage = self._dropout(self._modeling_layer(final_merged_passage, passage_lstm_mask))
        modeling_dim = modeled_passage.size(-1)


        # Shape: (batch_size, passage_length, encoding_dim * 4 + modeling_dim))
        span_start_input = self._dropout(torch.cat([final_merged_passage, modeled_passage], dim=-1))
        # Shape: (batch_size, passage_length)
        span_start_logits = self._span_start_predictor(span_start_input).squeeze(-1)
        # Shape: (batch_size, passage_length)
        span_start_probs = util.masked_softmax(span_start_logits, passage_mask)

        # Shape: (batch_size, modeling_dim)
        span_start_representation = util.weighted_sum(modeled_passage, span_start_probs)
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
        encoded_span_end = self._dropout(self._span_end_encoder(span_end_representation,
                                                                passage_lstm_mask))
        # Shape: (batch_size, passage_length, encoding_dim * 4 + span_end_encoding_dim)
        span_end_input = self._dropout(torch.cat([final_merged_passage, encoded_span_end], dim=-1))
        span_end_logits = self._span_end_predictor(span_end_input).squeeze(-1)
        span_end_probs = util.masked_softmax(span_end_logits, passage_mask)
        span_start_logits = util.replace_masked_values(span_start_logits, passage_mask, -1e7)
        span_end_logits = util.replace_masked_values(span_end_logits, passage_mask, -1e7)
        # answer_len for masking
        answer_len = [len(elem['answer_texts']) for elem in metadata] if metadata is not None else []
        if answer_len:
            mask = torch.zeros((batch_size, max(answer_len), 2)).long()
            for index, length in enumerate(answer_len):
                mask[index, :length] = 1
        else:
            mask = None

        best_span, top_span_logits = self.get_best_span(span_start_logits, span_end_logits, answer_len)

        output_dict = {
                "passage_question_attention": passage_question_attention,
                "span_start_logits": span_start_logits,
                "span_start_probs": span_start_probs,
                "span_end_logits": span_end_logits,
                "span_end_probs": span_end_probs,
                "best_span": best_span,
                }

        # Compute the loss for training.
        if span_start is not None:
            span_start = span_start.squeeze(-1) #batch X max_answer_L
            span_end = span_end.squeeze(-1) #batch X max_answer_L

            # a batch_size x passage_length tensor with 1's indicating right
            # answer at that position/index
            span_start_pos = torch.zeros((batch_size, passage_length))
            span_end_pos = torch.zeros((batch_size, passage_length))

            for row_id, row in enumerate(span_start):
                for span_index in row:
                    span_index = span_index.data[0]
                    if span_index == -1:
                        break
                    span_start_pos[row_id][span_index] = 1

            for row_id, row in enumerate(span_end):
                for span_index in row:
                    span_index = span_index.data[0]
                    if span_index == -1:
                        break
                    span_end_pos[row_id][span_index] = 1

            span_start_ground = to_variable(span_start_pos) # batch x passage_len
            span_end_ground = to_variable(span_end_pos) # batch x passage_len

            # at this point, we have a 2 - 2d matrix for start, end respectively
            # each matrix has the index of the right answer set to 1

            flattened_start_pred = flatten_answer(span_start_logits, passage_mask)
            flattened_end_pred = flatten_answer(span_end_logits, passage_mask)
            flattened_start_ground = flatten_answer(span_start_ground, passage_mask)
            flattened_end_ground = flatten_answer(span_end_ground, passage_mask)

            loss = binary_cross_entropy_with_logits(flattened_start_pred, flattened_start_ground)
            loss += binary_cross_entropy_with_logits(flattened_end_pred, flattened_end_ground)

            """
            #TODO for better reporting only
            self._span_start_accuracy(flattened_start_pred, flattened_start_ground)
            self._span_end_accuracy(flattened_end_pred, flattened_end_ground)
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1), mask)
            """
            """
            # OLD CODE - ONLY REFERENCE
            # TODO answer padding needs to be ignored
            step = 0
            span_start_1D = span_start[ : , step:step + 1] #batch X 1 
            span_end_1D = span_end[ : , step:step + 1] #batch X 1 
            loss = nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start_1D.squeeze(-1))
            self._span_start_accuracy(span_start_logits, span_start_1D.squeeze(-1)) #TODO
            loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end_1D.squeeze(-1))
            self._span_end_accuracy(span_end_logits, span_end_1D.squeeze(-1)) #TODO
            # self._span_accuracy(best_span, torch.stack([span_start_1D, span_end_1D], -1))#TODO

            for step in range(1, span_start.size(1)):
                span_start_1D = span_start[ : , step:step + 1] #batch X 1 
                span_end_1D = span_end[ : , step:step + 1] #batch X 1 
                loss += nll_loss(util.masked_log_softmax(span_start_logits, passage_mask), span_start_1D.squeeze(-1), ignore_index=-1)
                self._span_start_accuracy(span_start_logits, span_start_1D.squeeze(-1)) #TODO
                loss += nll_loss(util.masked_log_softmax(span_end_logits, passage_mask), span_end_1D.squeeze(-1), ignore_index=-1)
                self._span_end_accuracy(span_end_logits, span_end_1D.squeeze(-1)) #TODO
                # self._span_accuracy(best_span, torch.stack([span_start_1D, span_end_1D], -1))#TODO
            self._span_accuracy(best_span, torch.stack([span_start, span_end], -1), mask)
            """
            output_dict["loss"] = loss

        pscores = top_span_logits[:, :, 0] # 40 X 12
        span_starts = top_span_logits[:, :, 1] # 40 X 12
        span_ends = top_span_logits[:, :, 2] # 40 X 12  
        best_span_starts = best_span[:, :, 0] # 40 X 12 # to check for -1

        lr_list = [] #TODO: Place this is in the right spot
        label = 0
        pscore = 0

        # Compute the EM and F1 on SQuAD and add the tokenized input to the output.
        if metadata is not None:
            output_dict['best_span_str'] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                best_span_strings = []
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
                passage_str = metadata[i]['original_passage']
                offsets = metadata[i]['token_offsets']
                predicted_spans = tuple(best_span[i].data.cpu().numpy())
                for predicted_span in predicted_spans:
                    if predicted_span[0] == -1:
                        break
                    start_offset = offsets[predicted_span[0]][0]
                    end_offset = offsets[predicted_span[1]][1]
                    best_span_string = passage_str[start_offset:end_offset]
                    best_span_strings.append(best_span_string)
                output_dict['best_span_str'].append(best_span_strings)
                answer_texts = metadata[i].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_strings, answer_texts)
                for j in range(span_starts.shape[1]):
                    pscore = pscores.data[i][j]
                    if best_span_starts.data[i][j] != -1:
                        label = 1
                    else:
                        label = 0
                    
                    question_comp = metadata[i]['qID'].split(',')[1].replace('@', '-') #TODO: COnsidering only 1 question entity, what if no entity
                    answer_comp = passage_str[int(span_starts.data[i][j]) : int(span_ends.data[i][j])] #TODO: this will need some further processing
                    dijkstra_comp = metadata[i]['dijkstra']
                    import pdb; pdb.set_trace()
                    dscore = dijkstra_comp[question_comp][answer_comp] if question_comp in dijkstra_comp and answer_comp in dijkstra_comp[question_comp] else None
                    if dscore is not None:
                        lr_list.append((pscore, dscore, label))
                    
            output_dict['question_tokens'] = question_tokens
            output_dict['passage_tokens'] = passage_tokens

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'start_acc': 0.007, #self._span_start_accuracy.get_metric(reset),
                'end_acc': 0.007, #self._span_end_accuracy.get_metric(reset),
                'span_acc': 0.007, #self._span_accuracy.get_metric(reset),
                'em': exact_match,
                'f1': f1_score,
                }

    @staticmethod
    def get_best_span(span_start_logits: Variable, span_end_logits: Variable, answer_len: List[int] = []) -> (Variable,
            Variable):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        #span_start_logits, span_end_logits are both 40 X 707
        batch_size, passage_length = span_start_logits.size()
        max_span_log_prob = [-1e20] * batch_size # 40 size list
        span_start_argmax = [0] * batch_size #40 size list
        max_answer_length = max(answer_len) if len(answer_len) != 0 else 0 #gives out value 12

        """
        best_word_span = Variable(span_start_logits.data.new()
                                  .resize_(batch_size, 2).fill_(0)).long() # (40 x ans_len x 2)
        """
        spanned_answer_length = max_answer_length if max_answer_length != 0 else 3 #gives out value 12
        best_word_span = to_variable(torch.zeros((batch_size, spanned_answer_length, 2)).fill_(-1).long()) #40 X 12 X 2
        top_word_span_with_logits = to_variable(torch.zeros((batch_size, spanned_answer_length, 3)).fill_(-1)) #40 X 12 X 3

        span_start_logits = span_start_logits.data.cpu().numpy() # (40 x passage_len)
        span_end_logits = span_end_logits.data.cpu().numpy() # (40 x passage_len)

        for b in range(batch_size):  # pylint: disable=invalid-name
            curr_passage_span_list = []

            curr_answer_span_len = answer_len[b] if len(answer_len) != 0 else 3
            for j in range(passage_length): # go through each byte in the passage
                val1 = span_start_logits[b, span_start_argmax[b]] # get the span_start_logit for highest start index so far
                if val1 < span_start_logits[b, j]: # compare the existing start logit with current start logit
                    span_start_argmax[b] = j # we found a better start index, update this index
                    val1 = span_start_logits[b, j] # update the val1

                start_index = span_start_argmax[b]
                end_index = j
                val2 = span_end_logits[b, j] # the end logit 

                curr_passage_span_list.append((val1 + val2, start_index, end_index))
                """
                if val1 + val2 > max_span_log_prob[b]: # 
                    best_word_span[b, 0] = span_start_argmax[b] # for the current batch, which is the best start_index
                    best_word_span[b, 1] = j # for the current batch, which is the best end_index
                    max_span_log_prob[b] = val1 + val2 # store history for what is the best span's log probability until now
                """

            curr_passage_span_list = sorted(curr_passage_span_list, key=lambda x: x[0], reverse=True)
            top_span_logits = curr_passage_span_list[:spanned_answer_length] #12 tuples of the form (number, index, index)
            curr_passage_span_list = curr_passage_span_list[:curr_answer_span_len]

            tensor = torch.from_numpy(np.array([(start, end) for _, start, end in curr_passage_span_list])).long()
            best_word_span[b, :curr_answer_span_len] = tensor

            tensor = torch.from_numpy(np.array(top_span_logits)) #12 X 3
            top_word_span_with_logits[b, :spanned_answer_length] = tensor

        return best_word_span, top_word_span_with_logits

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BidirectionalAttentionFlow':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        num_highway_layers = params.pop_int("num_highway_layers")
        phrase_layer = Seq2SeqEncoder.from_params(params.pop("phrase_layer"))
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        span_end_encoder = Seq2SeqEncoder.from_params(params.pop("span_end_encoder"))
        dropout = params.pop_float('dropout', 0.2)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        mask_lstms = params.pop_bool('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   num_highway_layers=num_highway_layers,
                   phrase_layer=phrase_layer,
                   attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer,
                   span_end_encoder=span_end_encoder,
                   dropout=dropout,
                   mask_lstms=mask_lstms,
                   initializer=initializer,
                   regularizer=regularizer)
