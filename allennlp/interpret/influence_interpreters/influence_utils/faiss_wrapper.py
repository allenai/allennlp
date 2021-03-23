import faiss
from typing import Optional
from overrides import overrides
import torch
from .faiss_utils import FAISSIndex
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask


class FAISSWrapper(FAISSIndex, Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 description: Optional[str] = "Flat",
                 index: Optional[faiss.Index] = None,
                 **kwargs):
        Model.__init__(self, vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        FAISSIndex.__init__(self, self._seq2vec_encoder.get_output_dim(), description, index)

    def extract_tokens_from_input(self, **inputs) -> TextFieldTensors:
        raise NotImplementedError

    def __call__(self, inputs) -> torch.Tensor:
        # TODO (Leo): this might not be general enough, e.g. use label as part of seq2vec
        # however, currently, this seems fine. Also, curious about how to
        tokens = self.extract_tokens_from_input(**inputs)
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)
        # TODO (Leo): similar generality issue here
        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        return embedded_text


class FAISSSnliWrapper(FAISSWrapper):
    @overrides
    def extract_tokens_from_input(self, tokens: TextFieldTensors, label: torch.IntTensor = None):
        return tokens
