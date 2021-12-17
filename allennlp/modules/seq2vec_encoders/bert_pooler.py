from typing import Optional, Dict, Any


import torch
import torch.nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder


@Seq2VecEncoder.register("bert_pooler")
class BertPooler(Seq2VecEncoder):
    """
    The pooling layer at the end of the BERT model. This returns an embedding for the
    [CLS] token, after passing it through a non-linear tanh activation; the non-linear layer
    is also part of the BERT model. If you want to use the pretrained BERT model
    to build a classifier and you want to use the AllenNLP token-indexer ->
    token-embedder -> seq2vec encoder setup, this is the Seq2VecEncoder to use.
    (For example, if you want to experiment with other embedding / encoding combinations.)

    Registered as a `Seq2VecEncoder` with name "bert_pooler".

    # Parameters

    pretrained_model : `Union[str, BertModel]`, required
        The pretrained BERT model to use. If this is a string,
        we will call `transformers.AutoModel.from_pretrained(pretrained_model)`
        and use that.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights that override the
        weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
        with `torch.save()`.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretraiend weights.
    requires_grad : `bool`, optional (default = `True`)
        If True, the weights of the pooler will be updated during training.
        Otherwise they will not.
    dropout : `float`, optional, (default = `0.0`)
        Amount of dropout to apply after pooling
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        pretrained_model: str,
        *,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        load_weights: bool = True,
        requires_grad: bool = True,
        dropout: float = 0.0,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        from allennlp.common import cached_transformers

        model = cached_transformers.get(
            pretrained_model,
            False,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            load_weights=load_weights,
            **(transformer_kwargs or {}),
        )

        self._dropout = torch.nn.Dropout(p=dropout)

        import copy

        self.pooler = copy.deepcopy(model.pooler)
        for param in self.pooler.parameters():
            param.requires_grad = requires_grad
        self._embedding_dim = model.config.hidden_size

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self, tokens: torch.Tensor, mask: torch.BoolTensor = None, num_wrapping_dims: int = 0
    ):
        pooler = self.pooler
        for _ in range(num_wrapping_dims):
            from allennlp.modules import TimeDistributed

            pooler = TimeDistributed(pooler)
        pooled = pooler(tokens)
        pooled = self._dropout(pooled)
        return pooled
