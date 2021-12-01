from overrides import overrides
import torch
from typing import List

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


@Seq2SeqEncoder.register("compose")
class ComposeEncoder(Seq2SeqEncoder):

    """This class can be used to compose several encoders in sequence.

    Among other things, this can be used to add a "pre-contextualizer" before a Seq2SeqEncoder.

    Registered as a `Seq2SeqEncoder` with name "compose".

    # Parameters

    encoders : `List[Seq2SeqEncoder]`, required.
        A non-empty list of encoders to compose. The encoders must match in bidirectionality.
    """

    def __init__(self, encoders: List[Seq2SeqEncoder]):
        super().__init__()
        self.encoders = encoders
        for idx, encoder in enumerate(encoders):
            self.add_module("encoder%d" % idx, encoder)

        # Compute bidirectionality.
        all_bidirectional = all(encoder.is_bidirectional() for encoder in encoders)
        any_bidirectional = any(encoder.is_bidirectional() for encoder in encoders)
        self.bidirectional = all_bidirectional

        if all_bidirectional != any_bidirectional:
            raise ValueError("All encoders need to match in bidirectionality.")

        if len(self.encoders) < 1:
            raise ValueError("Need at least one encoder.")

        last_enc = None
        for enc in encoders:
            if last_enc is not None and last_enc.get_output_dim() != enc.get_input_dim():
                raise ValueError("Encoder input and output dimensions don't match.")
            last_enc = enc

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        # Parameters

        inputs : `torch.Tensor`, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, timesteps).

        # Returns

        A tensor computed by composing the sequence of encoders.
        """
        for encoder in self.encoders:
            inputs = encoder(inputs, mask)
        return inputs

    @overrides
    def get_input_dim(self) -> int:
        return self.encoders[0].get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self.encoders[-1].get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return self.bidirectional
