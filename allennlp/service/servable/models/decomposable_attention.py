import os
from typing import Dict

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import DecomposableAttention
from allennlp.service.servable import Servable, JsonDict

@Servable.register('decomposable_attention')
class DecomposableAttentionServable(Servable):
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        premise_text = inputs["premise"]
        hypothesis_text = inputs["hypothesis"]

        premise = TextField(self.tokenizer.tokenize(premise_text), token_indexers=self.token_indexers)
        hypothesis = TextField(self.tokenizer.tokenize(hypothesis_text), token_indexers=self.token_indexers)

        output_dict = self.model.predict_entailment(premise, hypothesis)
        output_dict["label_probs"] = output_dict["label_probs"].tolist()

        return output_dict
