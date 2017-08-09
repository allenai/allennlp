from typing import Dict

from allennlp.common import Params, constants
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.squad import SquadReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.models import BidirectionalAttentionFlow
from allennlp.service.servable import Servable, JSONDict

class BidafServable(Servable):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 model: BidirectionalAttentionFlow) -> None:
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.model = model

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question = TextField(self.tokenizer.tokenize(question_text), token_indexers=self.token_indexers)
        passage = TextField(self.tokenizer.tokenize(passage_text) + [SquadReader.STOP_TOKEN],
                            token_indexers=self.token_indexers)

        output_dict = self.model.predict_span(question, passage)
        output_dict["span_start_probs"] = output_dict["span_start_probs"].tolist()
        output_dict["span_end_probs"] = output_dict["span_end_probs"].tolist()

        return output_dict

    @classmethod
    def from_params(cls, params: Params) -> 'BidafServable':
        glove_path = params.pop("glove_path")
        constants.GLOVE_PATH = glove_path

        tokenizer = Tokenizer.from_params(params.pop("tokenizer"))

        token_indexers = {}
        token_indexer_params = params.pop('token_indexers')
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)

        vocab_dir = params.pop('vocab_dir')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = params.pop("model")
        assert model_params.pop("type") == "bidaf"
        model = BidirectionalAttentionFlow.from_params(vocab, model_params)

        return BidafServable(tokenizer=tokenizer,
                             token_indexers=token_indexers,
                             model=model)
