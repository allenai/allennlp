import os
from typing import Dict

from allennlp.common import Params
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
        self.model.eval()

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question = TextField(self.tokenizer.tokenize(question_text), token_indexers=self.token_indexers)
        passage = TextField(self.tokenizer.tokenize(passage_text) + [SquadReader.STOP_TOKEN],
                            token_indexers=self.token_indexers)

        output_dict = self.model.predict_span(question, passage)

        # best_span is np.int64, we need to get pure python types so we can serialize them
        output_dict["best_span"] = [x.item() for x in output_dict["best_span"]]

        # similarly, the probability tensors must be converted to lists
        output_dict["span_start_probs"] = output_dict["span_start_probs"].tolist()
        output_dict["span_end_probs"] = output_dict["span_end_probs"].tolist()

        return output_dict

    @classmethod
    def from_config(cls, config: Params) -> 'BidafServable':
        dataset_reader_params = config["dataset_reader"]
        assert dataset_reader_params.pop('type') == 'squad'
        dataset_reader = SquadReader.from_params(dataset_reader_params)

        serialization_prefix = config['trainer']['serialization_prefix']
        vocab_dir = os.path.join(serialization_prefix, 'vocabulary')
        vocab = Vocabulary.from_files(vocab_dir)

        model_params = config["model"]
        assert model_params.pop("type") == "bidaf"
        model = BidirectionalAttentionFlow.from_params(vocab, model_params)

        # TODO(joelgrus) load weights

        # pylint: disable=protected-access
        return BidafServable(tokenizer=dataset_reader._tokenizer,
                             token_indexers=dataset_reader._token_indexers,
                             model=model)
        # pylint: enable=protected-access
