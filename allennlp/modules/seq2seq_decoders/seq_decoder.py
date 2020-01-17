from typing import Dict, Optional

import torch
from torch.nn import Module

from allennlp.common import Registrable
from allennlp.modules import Embedding


class SeqDecoder(Module, Registrable):
    """
    A `SeqDecoder` abstract class representing the entire decoder (embedding and neural network) of
    a Seq2Seq architecture.
    This is meant to be used with `allennlp.models.encoder_decoder.composed_seq2seq.ComposedSeq2Seq`.

    The implementation of this abstract class ideally uses a
    decoder neural net `allennlp.modules.seq2seq_decoders.decoder_net.DecoderNet` for decoding.

    The `default_implementation`
    `allennlp.modules.seq2seq_decoders.seq_decoder.auto_regressive_seq_decoder.AutoRegressiveSeqDecoder`
    covers most use cases. More likely that we will use the default implementation instead of creating a new
    implementation.

    # Parameters

    target_embedder : `Embedding`, required
        Embedder for target tokens. Needed in the base class to enable weight tying.
    """

    default_implementation = "auto_regressive_seq_decoder"

    def __init__(self, target_embedder: Embedding) -> None:
        super().__init__()
        self.target_embedder = target_embedder

    def get_output_dim(self) -> int:
        """
        The dimension of each timestep of the hidden state in the layer before final softmax.
        Needed to check whether the model is compatible for embedding-final layer weight tying.
        """
        raise NotImplementedError()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        The decoder is responsible for computing metrics using the target tokens.
        """
        raise NotImplementedError()

    def forward(
        self,
        encoder_out: Dict[str, torch.LongTensor],
        target_tokens: Optional[Dict[str, torch.LongTensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decoding from encoded states to sequence of outputs
        also computes loss if `target_tokens` are given.

        # Parameters

        encoder_out : `Dict[str, torch.LongTensor]`, required
            Dictionary with encoded state, ideally containing the encoded vectors and the
            source mask.
        target_tokens : `Dict[str, torch.LongTensor]`, optional
            The output of `TextField.as_array()` applied on the target `TextField`.

       """

        raise NotImplementedError()

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
            Post processing for converting raw outputs to prediction during inference.
            The composing models such `allennlp.models.encoder_decoders.composed_seq2seq.ComposedSeq2Seq`
            can call this method when `decode` is called.
        """
        raise NotImplementedError()
