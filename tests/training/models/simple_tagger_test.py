
import torch
from torch.autograd import Variable

from allennlp.models.simple_tagger import SimpleTagger
from allennlp.data.dataset_readers import SequenceTaggingDatasetReader
from allennlp.testing.test_case import AllenNlpTestCase
from allennlp.data.vocabulary import Vocabulary


class SimpleTaggerTest(AllenNlpTestCase):

    def setUp(self):
        super(SimpleTaggerTest, self).setUp()
        self.write_sequence_tagging_files()

        dataset = SequenceTaggingDatasetReader(self.TRAIN_FILE).read()
        vocab = Vocabulary.from_dataset(dataset)
        self.vocab = vocab
        dataset.index_instances(vocab)
        self.dataset = dataset

    def test_forward_pass_runs_correctly(self):
        training_arrays = self.dataset.as_arrays()
        print(training_arrays)
        training_arrays = {key:Variable(torch.Tensor(value)) for key, value in training_arrays.items()}

        model = SimpleTagger(self.vocab)
        return_dict = model(**training_arrays)
        print(return_dict)

