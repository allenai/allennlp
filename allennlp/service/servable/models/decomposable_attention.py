import os
from typing import Dict

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import DecomposableAttention
from allennlp.nn import InitializerApplicator
from allennlp.service.servable import Servable, JSONDict

class DecomposableAttentionServable(Servable):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 model: DecomposableAttention) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.model = model
        self.model.eval()

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        premise_text = inputs["premise"]
        hypothesis_text = inputs["hypothesis"]

        premise = TextField(self.tokenizer.tokenize(premise_text), token_indexers=self.token_indexers)
        hypothesis = TextField(self.tokenizer.tokenize(hypothesis_text), token_indexers=self.token_indexers)

        output_dict = self.model.predict_entailment(premise, hypothesis)
        output_dict["label_probs"] = output_dict["label_probs"].tolist()

        return output_dict

    @classmethod
    def from_config(cls, config: Params) -> 'DecomposableAttentionServable':
        dataset_reader_params = config.pop("dataset_reader")
        assert dataset_reader_params.pop("type") == "snli"
        dataset_reader = SnliReader.from_params(dataset_reader_params)

        serialization_prefix = config.pop('serialization_prefix')
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config.pop("model")
        assert model_params.pop("type") == "decomposable_attention"
        model = DecomposableAttention.from_params(vocab, model_params)

        # pylint: disable=protected-access
        return DecomposableAttentionServable(tokenizer=dataset_reader._tokenizer,
                                             token_indexers=dataset_reader._token_indexers,
                                             model=model)
        # pylint: enable=protected-access
