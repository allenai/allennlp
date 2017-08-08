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

        reader_params = Params({
                'token_indexers': {
                        'tokens': {
                                'type': 'single_id'
                                },
                        'token_characters': {
                                'type': 'characters'
                                }
                        }
                })
        dataset = SquadReader.from_params(reader_params).read('tests/fixtures/squad_example.json')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = {'tokens': SingleIdTokenIndexer(),
                               'token_characters': TokenCharactersIndexer()}

        self.model = BidirectionalAttentionFlow.from_params(self.vocab, Params({}))

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        tokenizer = WordTokenizer()

        question_text = inputs["question"]
        passage_text = inputs["passage"]

        question = TextField(tokenizer.tokenize(question_text), token_indexers=self.token_indexers)
        passage = TextField(tokenizer.tokenize(passage_text) + [SquadReader.STOP_TOKEN],
                            token_indexers=self.token_indexers)

        output_dict = self.model.predict_span(question, passage)
        output_dict["span_start_probs"] = output_dict["span_start_probs"].tolist()
        output_dict["span_end_probs"] = output_dict["span_end_probs"].tolist()

        return output_dict
