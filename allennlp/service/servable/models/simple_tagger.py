import codecs
import os
from tempfile import gettempdir

from allennlp.common import Params
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.service.servable import Servable, JSONDict

# Simple Tagger Model
class SimpleTaggerServable(Servable):
    def __init__(self):
        temp_dir = gettempdir()
        train_fn = os.path.join(temp_dir, 'train.txt')
        with codecs.open(train_fn, 'w', 'utf-8') as train_file:
            train_file.write('cats###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('dogs###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('snakes###N\tare###V\tanimals###N\t.###N\n')
            train_file.write('birds###N\tare###V\tanimals###N\t.###N\n')

        self.dataset = SequenceTaggingDatasetReader().read(train_fn)
        self.vocab = Vocabulary.from_dataset(self.dataset)
        self.dataset.index_instances(self.vocab)

        self.params = Params({
                "text_field_embedder": {
                        "tokens": {
                                "type": "embedding",
                                "embedding_dim": 5
                                }
                        },
                "hidden_size": 7,
                "num_layers": 2
                })
        self.model = SimpleTagger.from_params(self.vocab, self.params)
        self.tokenizer = WordTokenizer()

    def predict_json(self, inputs: JSONDict) -> JSONDict:
        sentence = inputs["input"]
        tokens = self.tokenizer.tokenize(sentence)
        text = TextField(tokens, token_indexers={"tokens": SingleIdTokenIndexer()})
        output = self.model.tag(text)

        # convert np array to serializable list
        output['class_probabilities'] = output['class_probabilities'].tolist()

        possible_tags = list(self.vocab.get_index_to_token_vocabulary("tags").values())
        return {'model_name': 'simple_tagger',
                'input': sentence,
                'output': output,
                'tokens': tokens,
                'possible_tags': possible_tags
               }
