from typing import Dict

from allennlp.common import Params, constants
from allennlp.data import Vocabulary
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

        initializer = InitializerApplicator()
        initializer(self.model)

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        premise_text = inputs["premise"]
        hypothesis_text = inputs["hypothesis"]

        premise = TextField(self.tokenizer.tokenize(premise_text), token_indexers=self.token_indexers)
        hypothesis = TextField(self.tokenizer.tokenize(hypothesis_text), token_indexers=self.token_indexers)

        output_dict = self.model.predict_entailment(premise, hypothesis)
        output_dict["label_probs"] = output_dict["label_probs"].tolist()

        return output_dict

    @classmethod
    def from_params(cls, params: Params) -> 'DecomposableAttentionServable':
        glove_path = params.pop('glove_path')
        constants.GLOVE_PATH = glove_path

        tokenizer = Tokenizer.from_params(params.pop("tokenizer"))

        token_indexers = {}
        token_indexer_params = params.pop('token_indexers')
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)

        vocab_dir = params.pop('vocab_dir')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = params.pop("model")
        assert model_params.pop("type") == "decomposable_attention"
        model = DecomposableAttention.from_params(vocab, model_params)

        return DecomposableAttentionServable(tokenizer=tokenizer,
                                             token_indexers=token_indexers,
                                             model=model)
