import torch

from allennlp.common import FromParams
from allennlp.modules.attention import Attention
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.util import apply_mask


class BiModalAttention(TransformerModule, FromParams):
    """
    Computes attention for two modalities, based on
    [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations
    for Vision-and-Language Tasks (Lu et al, 2019)]
    (https://api.semanticscholar.org/CorpusID:199453025).

    From the paper:

    "The keys and values from each modality are passed as input to the
    other modality’s multi-headed attention block. Consequentially, the
    attention block produces attention-pooled features for each modality
    conditioned on the other."

    For example, considering the case when the first modality is image,
    and the second modality is language, the module performs
    "image-conditioned language attention in the visual stream and
    language-conditioned image attention in the linguistic stream."

    # Parameters

    hidden_size1 : `int`
        The input hidden dim for the first modality.
    hidden_size2 : `int`
        The input hidden dim for the second modality.
    combined_hidden_size : `int`
        The output hidden dim for both modalities; it should be a multiple
        of `num_attention_heads`.
    num_attention_heads : `int`
        The number of attention heads.
    dropout1 : `float` (default = `0.0`)
        The dropout probability for the first modality stream.
    dropout2 : `float` (default = `0.0`)
        The dropout probability for the second modality stream.
    scoring_func1 : `str` (default = `scaled_dot_product`)
        The name of the attention-calculating function to be used for the first modality.
    scoring_func2 : `str` (default = `scaled_dot_product`)
        The name of the attention-calculating function to be used for the second modality.
        Eg. `additive`, `linear`, etc. For a complete list, please check :mod:`allennlp.modules.attention`.
    """

    def __init__(
        self,
        hidden_size1: int,
        hidden_size2: int,
        combined_hidden_size: int,
        num_attention_heads: int,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        scoring_func1: str = "scaled_dot_product",
        scoring_func2: str = "scaled_dot_product",
    ):
        super().__init__()
        if combined_hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (combined_hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(combined_hidden_size / num_attention_heads)

        # This is basically the `combined_hidden_size`, since we already ensure
        # that `combined_hidden_size` is divisible by `num_attention_heads`.
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # First modality:

        self.query1 = torch.nn.Linear(hidden_size1, self.all_head_size)
        self.key1 = torch.nn.Linear(hidden_size1, self.all_head_size)
        self.value1 = torch.nn.Linear(hidden_size1, self.all_head_size)

        self.scoring_func1 = scoring_func1
        if self.scoring_func1 in ["additive", "linear", "bilinear"]:
            self.attn1 = Attention.by_name(self.scoring_func1)(hidden_size1, hidden_size1)
        elif self.scoring_func1 == "scaled_dot_product":
            self.attn1 = Attention.by_name(self.scoring_func1)(self.attention_head_size, False)
        else:
            self.attn1 = Attention.by_name(self.scoring_func1)()

        self.dropout1 = torch.nn.Dropout(dropout1)

        # Second modality:

        self.query2 = torch.nn.Linear(hidden_size2, self.all_head_size)
        self.key2 = torch.nn.Linear(hidden_size2, self.all_head_size)
        self.value2 = torch.nn.Linear(hidden_size2, self.all_head_size)

        self.scoring_func2 = scoring_func2
        if self.scoring_func2 in ["additive", "linear", "bilinear"]:
            self.attn2 = Attention.by_name(self.scoring_func2)(hidden_size2, hidden_size2)
        elif self.scoring_func2 == "scaled_dot_product":
            self.attn2 = Attention.by_name(self.scoring_func2)(self.attention_head_size, False)
        else:
            self.attn2 = Attention.by_name(self.scoring_func2)()

        self.dropout2 = torch.nn.Dropout(dropout2)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        input_tensor1,
        input_tensor2,
        attention_mask1=None,
        attention_mask2=None,
        co_attention_mask=None,  # TODO: is this flag necessary?
        use_co_attention_mask=False,
    ):
        """
        # Parameters

        input_tensor1 : `torch.Tensor`
            Shape `batch_size x seq_len1 x hidden_dim1`
            where `seq_len1` can be the sequence length
            when the modality is text, or the number of
            regions when the modality is image.
        input_tensor2 : `torch.Tensor`
            Shape `batch_size x seq_len2 x hidden_dim2`
            where `seq_len2` can be the sequence length
            when the modality is text, or the number of
            regions when the modality is image.
        attention_mask1 : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len1`
        attention_mask : `torch.BoolTensor`, optional
            Shape `batch_size x seq_len2`
        co_attention_mask : `torch.Tensor`, optional
            Shape `batch_size x seq_len1 x seq_len2 x all_head_size`
            This mask is for cases when you already have some prior information
            about the interaction between the two modalities. For example,
            if you know which words correspond to which regions in the image,
            this mask can be applied to limit the attention given the bias.
        use_co_attention_mask : `bool`
            Whether to use co_attention_mask or not, default = `False`.
        """

        # for the first modality:
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self._transpose_for_scores(mixed_query_layer1)
        key_layer1 = self._transpose_for_scores(mixed_key_layer1)
        value_layer1 = self._transpose_for_scores(mixed_value_layer1)

        # for the second modality:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        query_layer2 = self._transpose_for_scores(mixed_query_layer2)
        key_layer2 = self._transpose_for_scores(mixed_key_layer2)
        value_layer2 = self._transpose_for_scores(mixed_value_layer2)

        # Conditioning the second modality on the first one.
        attention_scores1 = self.attn1(query_layer2, key_layer1.transpose(-1, -2))
        if attention_mask1 is not None:
            attention_scores1 = apply_mask(attention_scores1, attention_mask1)
        if use_co_attention_mask:
            attention_scores1 = apply_mask(attention_scores1, co_attention_mask.permute(0, 1, 3, 2))

        attention_probs1 = torch.nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Conditioning the first modality on the second one.
        attention_scores2 = self.attn2(query_layer1, key_layer2.transpose(-1, -2))
        # we can comment this line for single flow.
        if attention_mask2 is not None:
            attention_scores2 = apply_mask(attention_scores2, attention_mask2)
        if use_co_attention_mask:
            attention_scores2 = apply_mask(attention_scores2, co_attention_mask)

        attention_probs2 = torch.nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        return context_layer1, context_layer2
