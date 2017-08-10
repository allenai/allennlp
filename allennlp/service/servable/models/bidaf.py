import os
from typing import Dict

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.squad import SquadReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
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
    def from_config(cls, config: Params) -> 'BidafServable':
        tokenizer_params = config.pop("tokenizer", None)
        if tokenizer_params is None:
            tokenizer = WordTokenizer()
        else:
            tokenizer = Tokenizer.from_params(tokenizer_params)

        token_indexers = {}
        token_indexer_params = config["dataset_reader"].pop('token_indexers')
        for name, indexer_params in token_indexer_params.items():
            token_indexers[name] = TokenIndexer.from_params(indexer_params)

        serialization_prefix = config.pop('serialization_prefix')
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config.pop("model")
        assert model_params.pop("type") == "bidaf"
        model = BidirectionalAttentionFlow.from_params(vocab, model_params)

        # TODO(joelgrus) load weights

        return BidafServable(tokenizer=tokenizer,
                             token_indexers=token_indexers,
                             model=model)
