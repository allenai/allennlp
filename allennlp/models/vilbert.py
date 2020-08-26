import collections
import logging
import math
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common import FromParams
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy

from transformers.modeling_auto import AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_bert import ACT2FN

logger = logging.getLogger(__name__)


class BertEmbeddings(torch.nn.Module, FromParams):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int,
        max_position_embeddings: int,
        type_vocab_size: int,
        dropout: float,
    ):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in
        # BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class BertSelfOutput(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, attention_dropout)
        self.output = BertSelfOutput(hidden_size, hidden_dropout)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, intermediate_size)
        if isinstance(activation, str):
            self.intermediate_act_fn = ACT2FN[activation]
        else:
            self.intermediate_act_fn = activation

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        attention_dropout: float,
        hidden_dropout: float,
        activation: str,
    ):
        super().__init__()
        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.intermediate = BertIntermediate(
            hidden_size=hidden_size, intermediate_size=intermediate_size, activation=activation
        )
        self.output = BertOutput(
            hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=hidden_dropout
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertBiAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        num_attention_heads: int,
        dropout1: float,
        dropout2: float,
    ):
        super().__init__()
        if combined_hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (combined_hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(combined_hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query1 = torch.nn.Linear(hidden_size1, self.all_head_size)
        self.key1 = torch.nn.Linear(hidden_size1, self.all_head_size)
        self.value1 = torch.nn.Linear(hidden_size1, self.all_head_size)

        self.dropout1 = torch.nn.Dropout(dropout1)

        self.query2 = torch.nn.Linear(hidden_size2, self.all_head_size)
        self.key2 = torch.nn.Linear(hidden_size2, self.all_head_size)
        self.value2 = torch.nn.Linear(hidden_size2, self.all_head_size)

        self.dropout2 = torch.nn.Dropout(dropout2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw
        # attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = torch.nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the
        # raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = torch.nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        return context_layer1, context_layer2


class BertBiOutput(torch.nn.Module):
    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        dropout1: float,
        dropout2: float,
    ):
        super().__init__()

        self.dense1 = torch.nn.Linear(combined_hidden_size, hidden_size1)
        self.layer_norm1 = torch.nn.LayerNorm(hidden_size1, eps=1e-12)
        self.dropout1 = torch.nn.Dropout(dropout1)

        self.dense2 = torch.nn.Linear(combined_hidden_size, hidden_size2)
        self.layer_norm2 = torch.nn.LayerNorm(hidden_size2, eps=1e-12)
        self.dropout2 = torch.nn.Dropout(dropout2)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.layer_norm1(context_state1 + input_tensor1)
        hidden_states2 = self.layer_norm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        intermediate_size1: int,
        intermediate_size2: int,
        num_attention_heads: int,
        dropout1: float,
        dropout2: float,
        activation: str,
    ):
        super().__init__()
        self.biattention = BertBiAttention(
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            combined_hidden_size=combined_hidden_size,
            num_attention_heads=num_attention_heads,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        self.biOutput = BertBiOutput(
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            combined_hidden_size=combined_hidden_size,
            dropout1=dropout1,
            dropout2=dropout2,
        )

        self.v_intermediate = BertIntermediate(
            hidden_size=hidden_size1,
            intermediate_size=intermediate_size1,
            activation=activation,
        )
        self.v_output = BertOutput(
            hidden_size=hidden_size1,
            intermediate_size=intermediate_size1,
            dropout=dropout1,
        )

        self.t_intermediate = BertIntermediate(
            hidden_size=hidden_size2,
            intermediate_size=intermediate_size2,
            activation=activation,
        )
        self.t_output = BertOutput(
            hidden_size=hidden_size2,
            intermediate_size=intermediate_size2,
            dropout=dropout2,
        )

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2 = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2


class BertEncoder(torch.nn.Module, FromParams):
    def __init__(
        self,
        text_num_hidden_layers: int,
        image_num_hidden_layers: int,
        text_hidden_size: int,
        image_hidden_size: int,
        combined_hidden_size: int,
        text_intermediate_size: int,
        image_intermediate_size: int,
        num_attention_heads: int,
        text_attention_dropout: float,
        text_hidden_dropout: float,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        activation: str,
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
    ):
        super().__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = fast_mode
        self.with_coattention = with_coattention
        self.v_biattention_id = v_biattention_id
        self.t_biattention_id = t_biattention_id
        self.in_batch_pairs = in_batch_pairs
        self.fixed_t_layer = fixed_t_layer
        self.fixed_v_layer = fixed_v_layer
        self.combined_size = combined_hidden_size
        self.text_hidden_size = text_hidden_size
        self.image_hidden_size = image_hidden_size

        layer = BertLayer(
            hidden_size=text_hidden_size,
            intermediate_size=text_intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=text_attention_dropout,
            hidden_dropout=text_hidden_dropout,
            activation=activation,
        )
        v_layer = BertLayer(
            hidden_size=image_hidden_size,
            intermediate_size=image_intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=image_attention_dropout,
            hidden_dropout=image_hidden_dropout,
            activation=activation,
        )
        connect_layer = BertConnectionLayer(
            hidden_size1=text_hidden_size,
            hidden_size2=image_hidden_size,
            combined_hidden_size=combined_hidden_size,
            intermediate_size1=text_intermediate_size,
            intermediate_size2=image_intermediate_size,
            num_attention_heads=num_attention_heads,
            dropout1=text_hidden_dropout,
            dropout2=image_hidden_dropout,
            activation=activation,
        )

        self.layer = torch.nn.ModuleList([deepcopy(layer) for _ in range(text_num_hidden_layers)])
        self.v_layer = torch.nn.ModuleList(
            [deepcopy(v_layer) for _ in range(image_num_hidden_layers)]
        )
        self.c_layer = torch.nn.ModuleList(
            [deepcopy(connect_layer) for _ in range(len(v_biattention_id))]
        )

    @classmethod
    def from_huggingface_model(
        cls,
        model: PreTrainedModel,
        image_num_hidden_layers: int,
        image_hidden_size: int,
        combined_hidden_size: int,
        image_intermediate_size: int,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
    ):
        config = model.config

        # TODO(mattg): this is brittle and will only work for particular transformer models.  But,
        # it's the best we can do for now, I think.
        num_attention_heads = config.num_attention_heads
        text_num_hidden_layers = config.num_hidden_layers
        text_hidden_size = config.hidden_size
        text_intermediate_size = config.intermediate_size
        text_attention_dropout = config.attention_probs_dropout_prob
        text_hidden_dropout = config.hidden_dropout_prob
        activation = config.hidden_act
        encoder = cls(
            text_num_hidden_layers=text_num_hidden_layers,
            image_num_hidden_layers=image_num_hidden_layers,
            text_hidden_size=text_hidden_size,
            image_hidden_size=image_hidden_size,
            combined_hidden_size=combined_hidden_size,
            text_intermediate_size=text_intermediate_size,
            image_intermediate_size=image_intermediate_size,
            num_attention_heads=num_attention_heads,
            text_attention_dropout=text_attention_dropout,
            text_hidden_dropout=text_hidden_dropout,
            image_attention_dropout=image_attention_dropout,
            image_hidden_dropout=image_hidden_dropout,
            activation=activation,
            v_biattention_id=v_biattention_id,
            t_biattention_id=t_biattention_id,
            fixed_t_layer=fixed_t_layer,
            fixed_v_layer=fixed_v_layer,
            fast_mode=fast_mode,
            with_coattention=with_coattention,
            in_batch_pairs=in_batch_pairs,
        )

        # After creating the encoder, we copy weights over from the transformer.  This currently
        # requires that the internal structure of the text side of this encoder *exactly matches*
        # the internal structure of whatever transformer you're using.  This could be made more
        # general by having some weight name transforms, or a name mapping, or something.  If we do
        # this, making it a class like NameMapper would probably be good, because specifying the
        # options without having a class to do it seems like a mess.
        encoder_parameters = dict(encoder.named_parameters())
        for name, parameter in model.named_parameters():
            if name.startswith("encoder."):
                name = name[8:]
                name = name.replace("LayerNorm", "layer_norm")
                if name not in encoder_parameters:
                    raise ValueError(
                        f"Couldn't find a matching parameter for {name}. Is this transformer "
                        "compatible with the joint encoder you're using?"
                    )
                encoder_parameters[name].data.copy_(parameter.data)

        return encoder

    def forward(
        self,
        txt_embedding,
        image_embedding,
        txt_attention_mask,
        image_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):
            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding = self.layer[idx](txt_embedding, txt_attention_mask)
                    t_start = self.fixed_t_layer

            for idx in range(t_start, t_end):
                txt_embedding = self.layer[idx](txt_embedding, txt_attention_mask)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding = self.v_layer[idx](image_embedding, image_attention_mask)
                    v_start = self.fixed_v_layer

            for idx in range(v_start, v_end):
                image_embedding = self.v_layer[idx](image_embedding, image_attention_mask)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                    .expand(batch_size, batch_size, num_regions, v_hidden_size)
                    .contiguous()
                    .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                    .expand(batch_size, batch_size, 1, 1, num_regions)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                    .expand(batch_size, batch_size, num_words, t_hidden_size)
                    .contiguous()
                    .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                    .expand(batch_size, batch_size, 1, 1, num_words)
                    .contiguous()
                    .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                    .expand(batch_size, batch_size, 1, num_regions, num_words)
                    .contiguous()
                    .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                txt_embedding, image_embedding = self.c_layer[count](
                    txt_embedding,
                    txt_attention_mask,
                    image_embedding,
                    image_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding = self.v_layer[idx](image_embedding, image_attention_mask)

        for idx in range(t_start, len(self.layer)):
            txt_embedding = self.layer[idx](txt_embedding, txt_attention_mask)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            torch.stack(all_encoder_layers_t, dim=-1),
            torch.stack(all_encoder_layers_v, dim=-1),
        )


class BertPooler(torch.nn.Module, FromParams):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImageFeatureEmbeddings(torch.nn.Module, FromParams):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.image_embeddings = torch.nn.Linear(feature_dim, hidden_dim)
        self.image_location_embeddings = torch.nn.Linear(4, hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, image_feature: torch.Tensor, image_location: torch.Tensor):
        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)
        embeddings = self.layer_norm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


@Model.register("nlvr2_vilbert")
@Model.register("nlvr2_vilbert_from_huggingface", constructor="from_huggingface_model_name")
class Nlvr2Vilbert(Model):
    """
    Model for the NLVR2 task based on the LXMERT paper (Tan et al. 2019).
    Parameters
    ----------
    vocab: ``Vocabulary``
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_embeddings: BertEmbeddings,
        image_embeddings: BertImageFeatureEmbeddings,
        encoder: BertEncoder,
        pooled_output_dim: int,
        fusion_method: str = "sum",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vocab)
        self.loss = torch.nn.CrossEntropyLoss()
        self.consistency_wrong_map: Dict[str, int] = collections.Counter()
        self._denotation_accuracy = CategoricalAccuracy()
        self.fusion_method = fusion_method

        self.embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.encoder = TimeDistributed(encoder)

        self.t_pooler = TimeDistributed(BertPooler(encoder.text_hidden_size, pooled_output_dim))
        self.v_pooler = TimeDistributed(BertPooler(encoder.image_hidden_size, pooled_output_dim))
        self.classifier = torch.nn.Linear(pooled_output_dim * 2, 2)
        self.dropout = torch.nn.Dropout(dropout)

    @classmethod
    def from_huggingface_model_name(
        cls,
        vocab: Vocabulary,
        model_name: str,
        image_feature_dim: int,
        image_num_hidden_layers: int,
        image_hidden_size: int,
        combined_hidden_size: int,
        pooled_output_dim: int,
        image_intermediate_size: int,
        image_attention_dropout: float,
        image_hidden_dropout: float,
        v_biattention_id: List[int],
        t_biattention_id: List[int],
        fixed_t_layer: int,
        fixed_v_layer: int,
        pooled_dropout: float = 0.1,
        fusion_method: str = "sum",
        fast_mode: bool = False,
        with_coattention: bool = True,
        in_batch_pairs: bool = False,
    ):
        transformer = AutoModel.from_pretrained(model_name)

        # TODO(mattg): This call to `transformer.embeddings` works with some transformers, but I'm
        # not sure it works for all of them, or what to do if it fails.
        # We should probably pull everything up until the instantiation of the image feature
        # embedding out into a central "transformers_util" module, or something, and just have a
        # method that pulls an initialized embedding layer out of a huggingface model.  One place
        # for this somewhat hacky code to live, instead of having to duplicate it in various models.
        text_embeddings = deepcopy(transformer.embeddings)

        # Albert (and maybe others?) has this "embedding_size", that's different from "hidden_size".
        # To get them to the same dimensionality, it uses a linear transform after the embedding
        # layer, which we need to pull out and copy here.
        if hasattr(transformer.config, "embedding_size"):
            config = transformer.config

            from transformers.modeling_albert import AlbertModel

            if isinstance(transformer, AlbertModel):
                linear_transform = deepcopy(transformer.encoder.embedding_hidden_mapping_in)
            else:
                logger.warning(
                    "Unknown model that uses separate embedding size; weights of the linear "
                    f"transform will not be initialized.  Model type is: {transformer.__class__}"
                )
                linear_transform = torch.nn.Linear(config.embedding_dim, config.hidden_dim)

            # We can't just use torch.nn.Sequential here, even though that's basically all this is,
            # because Sequential doesn't accept *inputs, only a single argument.

            class EmbeddingsShim(torch.nn.Module):
                def __init__(self, embeddings: torch.nn.Module, linear_transform: torch.nn.Module):
                    super().__init__()
                    self.linear_transform = linear_transform
                    self.embeddings = embeddings

                def forward(self, *inputs, **kwargs):
                    return self.linear_transform(self.embeddings(*inputs, **kwargs))

            text_embeddings = EmbeddingsShim(text_embeddings, linear_transform)

        image_embeddings = BertImageFeatureEmbeddings(
            feature_dim=image_feature_dim,
            hidden_dim=image_hidden_size,
            dropout=image_hidden_dropout,
        )
        encoder = BertEncoder.from_huggingface_model(
            model=transformer,
            image_num_hidden_layers=image_num_hidden_layers,
            image_hidden_size=image_hidden_size,
            combined_hidden_size=combined_hidden_size,
            image_intermediate_size=image_intermediate_size,
            image_attention_dropout=image_attention_dropout,
            image_hidden_dropout=image_hidden_dropout,
            v_biattention_id=v_biattention_id,
            t_biattention_id=t_biattention_id,
            fixed_t_layer=fixed_t_layer,
            fixed_v_layer=fixed_v_layer,
            fast_mode=fast_mode,
            with_coattention=with_coattention,
            in_batch_pairs=in_batch_pairs,
        )
        return cls(
            vocab=vocab,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            encoder=encoder,
            pooled_output_dim=pooled_output_dim,
            fusion_method=fusion_method,
            dropout=pooled_dropout,
        )

    def consistency(self, reset: bool) -> float:
        num_consistent_groups = sum(1 for c in self.consistency_wrong_map.values() if c == 0)
        value = float(num_consistent_groups) / len(self.consistency_wrong_map)
        if reset:
            self.consistency_wrong_map.clear()
        return value

    @overrides
    def forward(
        self,  # type: ignore
        sentence: List[str],
        visual_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        image_id: List[List[str]],
        identifier: List[str],
        sentence_field: TextFieldTensors,
        denotation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, num_images, _, feature_size = visual_features.size()

        input_ids = sentence_field["tokens"]["token_ids"]
        token_type_ids = sentence_field["tokens"]["type_ids"]
        attention_mask = sentence_field["tokens"]["mask"]
        # All batch instances will always have the same number of images and boxes, so no masking
        # is necessary, and this is just a tensor of ones.
        image_attention_mask = torch.ones_like(box_coordinates[:, :, :, 0])

        # (batch_size, num_tokens, embedding_dim)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        num_tokens = embedding_output.size(1)

        # Repeat the embedding dimension, so that the TimeDistributed works out ok
        embedding_output = embedding_output.unsqueeze(1).expand(-1, 2, -1, -1)
        attention_mask = attention_mask.unsqueeze(1).expand(-1, 2, -1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of
        # causal attention used in OpenAI GPT, we just need to prepare the
        # broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(2).unsqueeze(3).float().log()
        extended_image_attention_mask = image_attention_mask.unsqueeze(2).unsqueeze(3).float().log()

        # TODO(matt): it looks like the co-attention logic is all currently commented out; not sure
        # that this is necessary.
        extended_co_attention_mask = torch.zeros(
            batch_size,
            num_images,
            1,
            feature_size,
            num_tokens,
            dtype=extended_image_attention_mask.dtype,
        )

        # (batch_size, num_images, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(visual_features, box_coordinates)
        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        sequence_output_t = encoded_layers_t[:, :, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, :, -1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            raise ValueError(f"Fusion method '{self.fusion_method}' not supported")

        hidden_dim = pooled_output.size(-1)
        logits = self.classifier(pooled_output.view(batch_size, num_images * hidden_dim))

        outputs = {}
        outputs["logits"] = logits
        if denotation is not None:
            outputs["loss"] = self.loss(logits, denotation).sum()
            self._denotation_accuracy(logits, denotation)
            # Update group predictions for consistency computation
            predicted_binary = logits.argmax(1)
            for i in range(len(identifier)):
                ident_parts = identifier[i].split("-")
                group_id = "-".join([ident_parts[0], ident_parts[1], ident_parts[-1]])
                self.consistency_wrong_map.setdefault(group_id, 0)
                if predicted_binary[i].item() != denotation[i].item():
                    self.consistency_wrong_map[group_id] += 1
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
