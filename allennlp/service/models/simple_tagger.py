from typing import Callable, Dict

from allennlp.common import Params
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.service.models.types import Model, JSON


# Simple Tagger Model
def simple_tagger_model() -> Model:
    """create a simple tagger model."""
    # this is a bad hack to get the same data as the test case
    # TODO(joelgrus): replace this
    test_case = AllenNlpTestCase()
    test_case.setUp()
    test_case.write_sequence_tagging_data()
    dataset = SequenceTaggingDatasetReader().read(test_case.TRAIN_FILE)

    vocab = Vocabulary.from_dataset(dataset)
    dataset.index_instances(vocab)

    params = Params({
            "text_field_embedder": {
                    "tokens": {
                            "type": "embedding",
                            "embedding_dim": 5
                            }
                    },
            "stacked_encoder": {
                    "type": "lstm",
                    "input_size": 5,
                    "hidden_size": 7,
                    "num_layers": 2
                    }
            })
    model = SimpleTagger.from_params(vocab, params)
    tokenizer = WordTokenizer()

    def run(blob: JSON):
        sentence = blob.get("input", "")
        tokens = tokenizer.tokenize(sentence)
        text = TextField(tokens, token_indexers={"tokens": SingleIdTokenIndexer()})
        output = model.tag(text)

        # convert np array to serializable list
        output['class_probabilities'] = output['class_probabilities'].tolist()

        possible_tags = list(vocab.get_index_to_token_vocabulary("tags").values())
        return {'model_name': 'simple_tagger',
                'input': sentence,
                'output': output,
                'tokens': tokens,
                'possible_tags': possible_tags
               }

    return run

def models() -> Dict[str, Callable[[], Model]]:
    return {'simple_tagger': simple_tagger_model}
