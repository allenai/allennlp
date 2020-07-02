import logging
import math
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from transformers import AutoConfig
from transformers.modeling_bert import (
    ACT2FN,
    BertEmbeddings,
    BertIntermediate,
    BertLayerNorm,
    BertOutput,
    BertSelfOutput,
)

logger = logging.getLogger(__name__)


class BertSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
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

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(config.v_hidden_size / config.v_num_attention_heads)

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = torch.nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = torch.nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = torch.nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = torch.nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the
        # raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel
        # forward() function)
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

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertImageSelfOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(
            input_tensor, attention_mask, txt_embedding, txt_attention_mask
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = torch.nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(config.bi_hidden_size / config.bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = torch.nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = torch.nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = torch.nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = torch.nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = torch.nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = torch.nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = torch.nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = torch.nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
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

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense1 = torch.nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = torch.nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = torch.nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = torch.nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = torch.nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = torch.nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = torch.nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):

        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        co_attention_mask=None,
        use_co_attention_mask=False,
    ):

        bi_output1, bi_output2, co_attention_probs = self.biattention(
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

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = torch.nn.ModuleList([deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.v_layer = torch.nn.ModuleList(
            [deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )
        self.c_layer = torch.nn.ModuleList(
            [deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

    def forward(
        self,
        txt_embedding,
        image_embedding,
        txt_attention_mask,
        txt_attention_mask2,
        image_attention_mask,
        co_attention_mask=None,
        output_all_encoded_layers=True,
        output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attention_mask_v = []
        all_attention_mask_c = []

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
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attention_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attention_mask_v.append(image_attention_probs)

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
                    image_embedding.size(0), txt_embedding.size(1), txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[count](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding, image_attention_mask, txt_embedding, txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attention_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](txt_embedding, txt_attention_mask)

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            torch.stack(all_encoder_layers_t, dim=-1),
            torch.stack(all_encoder_layers_v, dim=-1),
        )


class BertTextPooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = torch.nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImageFeatureEmbeddings(torch.nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and
    token_type embeddings.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.image_embeddings = torch.nn.Linear(feature_dim, hidden_dim)
        self.image_location_embeddings = torch.nn.Linear(4, hidden_dim)
        self.layer_norm = BertLayerNorm(hidden_dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, image_feature: torch.Tensor, image_location: torch.Tensor):
        img_embeddings = self.image_embeddings(image_feature)
        loc_embeddings = self.image_location_embeddings(image_location)
        embeddings = self.layer_norm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


@Model.register("nlvr2_vilbert")
class Nlvr2Vilbert(Model):
    """
    Model for the NLVR2 task based on the LXMERT paper (Tan et al. 2019).
    Parameters
    ----------
    vocab: ``Vocabulary``
    """

    def __init__(
        self, vocab: Vocabulary, image_feature_dim: int, image_hidden_dim: int, fusion_method: str
    ) -> None:
        super().__init__(vocab)
        self.loss = torch.nn.CrossEntropyLoss()
        self.consistency_wrong_map = {}
        self._denotation_accuracy = CategoricalAccuracy()
        self.fusion_method = fusion_method
        bert_config = AutoConfig.from_pretrained("bert-base-uncased")

        # initilize word embedding
        self.embeddings = BertEmbeddings(bert_config)

        # initlize the vision embedding
        self.image_embeddings = BertImageFeatureEmbeddings(
            image_feature_dim, image_hidden_dim, bert_config.hidden_dropout_prob
        )

        bert_config.fast_mode = False
        bert_config.with_coattention = True
        bert_config.v_biattention_id = [0, 1, 2, 3, 4, 5]
        bert_config.t_biattention_id = [6, 7, 8, 9, 10, 11]
        bert_config.in_batch_pairs = False
        bert_config.fixed_t_layer = 0
        bert_config.fixed_v_layer = 0
        bert_config.num_hidden_layers = 12
        bert_config.v_num_hidden_layers = 6
        bert_config.visualization = False
        bert_config.v_hidden_size = 1024
        bert_config.v_num_attention_heads = 8
        bert_config.v_intermediate_size = 1024
        bert_config.v_attention_probs_dropout_prob = 0.1
        bert_config.v_hidden_dropout_prob = 0.1
        bert_config.bi_hidden_size = 1024
        bert_config.bi_num_attention_heads = 8
        bert_config.bi_intermediate_size = 1024
        bert_config.dynamic_attention = False
        bert_config.v_hidden_act = "gelu"
        self.encoder = TimeDistributed(BertEncoder(bert_config))
        self.t_pooler = TimeDistributed(BertTextPooler(bert_config))
        self.v_pooler = TimeDistributed(BertImagePooler(bert_config))
        self.dropout = torch.nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(bert_config.bi_hidden_size * 2, 2)

    def consistency(self, reset: bool) -> float:
        num_consistent_groups = 0
        for key in self.consistency_wrong_map:
            if self.consistency_wrong_map[key] == 0:
                num_consistent_groups += 1
        value = float(num_consistent_groups) / len(self.consistency_wrong_map)
        if reset:
            self.consistency_wrong_map = {}
        return value

    def forward(
        self,
        sentence: List[str],
        visual_features: torch.Tensor,
        box_coordinates: torch.Tensor,
        image_id: List[List[str]],
        identifier: List[str],
        sentence_field: TextFieldTensors,
        denotation: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        batch_size, num_images, num_objects, feature_size = visual_features.size()
        assert num_objects == 36 and feature_size == 2048
        sentence_mask = util.get_text_field_mask(sentence_field).float()
        assert num_images == 2

        input_ids = sentence_field["tokens"]["token_ids"]
        token_type_ids = sentence_field["tokens"]["type_ids"]
        attention_mask = sentence_field["tokens"]["mask"]
        # We will always have an equal number of images and boxes, so no masking necessary.
        image_attention_mask = torch.ones_like(box_coordinates)[:, :, :, 0]

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
        extended_attention_mask = attention_mask.unsqueeze(2).unsqueeze(3)
        extended_image_attention_mask = image_attention_mask.unsqueeze(2).unsqueeze(3)

        extended_attention_mask2 = attention_mask.unsqueeze(3)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        co_attention_mask = torch.zeros(
            batch_size, num_images, feature_size, num_tokens
        ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(2)

        # (batch_size, num_images, num_boxes, image_embedding_dim)
        v_embedding_output = self.image_embeddings(visual_features, box_coordinates)
        print(embedding_output.size())
        print(v_embedding_output.size())
        print(extended_attention_mask.size())
        print(extended_attention_mask2.size())
        print(extended_image_attention_mask.size())
        print(extended_co_attention_mask.size())
        print("ENCODER")
        encoded_layers_t, encoded_layers_v = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
        )

        print(encoded_layers_t.size())
        print(encoded_layers_v.size())
        sequence_output_t = encoded_layers_t[:, :, :, :, -1]
        sequence_output_v = encoded_layers_v[:, :, :, :, -1]

        print("POOLER")
        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)

        print("CLASSIFER")
        print(pooled_output.size())
        hidden_dim = pooled_output.size(-1)
        logits = self.classifier(pooled_output.view(batch_size, num_images * hidden_dim))
        print(logits.size())
        print(denotation.size())

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
                if group_id not in self.consistency_wrong_map:
                    self.consistency_wrong_map[group_id] = 0
                if predicted_binary[i].item() != denotation[i].item():
                    self.consistency_wrong_map[group_id] += 1
        return outputs

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "denotation_acc": self._denotation_accuracy.get_metric(reset),
            "consistency": self.consistency(reset),
        }
