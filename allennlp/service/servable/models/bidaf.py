from allennlp.common import Params, constants
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.squad import SquadReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.models import BidirectionalAttentionFlow
from allennlp.service.servable import Servable, JSONDict

class BidafServable(Servable):
    def __init__(self):
        constants.GLOVE_PATH = 'tests/fixtures/glove.6B.100d.sample.txt.gz'

        self.token_indexers = {'tokens': SingleIdTokenIndexer(lowercase_tokens=True),
                               'token_characters': TokenCharactersIndexer()}

        self.tokenizer = WordTokenizer()

        dataset = SquadReader(token_indexers=self.token_indexers).read('tests/fixtures/squad_example.json')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset

        self.model = BidirectionalAttentionFlow.from_params(self.vocab, Params({}))

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
