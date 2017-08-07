from allennlp.common import Params, constants
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.models import DecomposableAttention
from allennlp.nn import InitializerApplicator
from allennlp.service.servable import Servable, JSONDict

class DecomposableAttentionServable(Servable):
    def __init__(self):
        constants.GLOVE_PATH = 'tests/fixtures/glove.6B.300d.sample.txt.gz'
        dataset = SnliReader().read('tests/fixtures/snli_example.jsonl')
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

        self.model = DecomposableAttention.from_params(self.vocab, Params({}))
        initializer = InitializerApplicator()
        initializer(self.model)


    def predict_json(self, inputs: JSONDict) -> JSONDict:
        tokenizer = WordTokenizer()

        # TODO(joelgrus) fix front-end to specify these
        if "premise" in inputs and "hypothesis" in inputs:
            premise_text = inputs["premise"]
            hypothesis_text = inputs["hypothesis"]
        else:
            premise_text, hypothesis_text = inputs["input"].split("\n\n")


        premise = TextField(tokenizer.tokenize(premise_text), token_indexers=self.token_indexers)
        hypothesis = TextField(tokenizer.tokenize(hypothesis_text), token_indexers=self.token_indexers)

        output_dict = self.model.predict_entailment(premise, hypothesis)
        output_dict["label_probs"] = output_dict["label_probs"].tolist()

        return output_dict
