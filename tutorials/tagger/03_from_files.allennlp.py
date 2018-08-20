# from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# pylint: disable=invalid-name,arguments-differ
from typing import Iterator, List
import itertools
import shutil
import tempfile

import torch

from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.trainer import Trainer

torch.manual_seed(1)

@DatasetReader.register('pos-tutorial')
class PosDatasetReader(DatasetReader):
    def __init__(self) -> None:
        super().__init__(lazy=False)
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: List[str], tags: List[str]) -> Instance:
        tokens = [Token(word) for word in sentence]
        sentence_field = TextField(tokens, self.token_indexers)
        label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        return Instance(fields={"sentence": sentence_field,
                                "labels": label_field})

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for separator, group in itertools.groupby(f, lambda line: line.strip() == ''):
                if not separator:
                    sentence, tags = zip(*[line.split() for line in group])
                    yield self.text_to_instance(sentence, tags)


@Model.register('lstm-tagger')
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = FeedForward(input_dim=encoder.get_output_dim(),
                                      num_layers=1,
                                      hidden_dims=vocab.get_vocab_size('labels'),
                                      activations=lambda x: x)

    def forward(self, sentence: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.word_embeddings(sentence)
        mask = get_text_field_mask(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

params = Params.from_file('tutorials/tagger/experiment.jsonnet')

reader = DatasetReader.from_params(params.pop("dataset_reader"))
instances = reader.read(params.pop('train_data_path'))
vocab = Vocabulary.from_instances(instances)

model = Model.from_params(params.pop('model'), vocab=vocab)
iterator = DataIterator.from_params(params.pop('iterator'))
iterator.index_with(vocab)

serialization_dir = tempfile.mkdtemp()
trainer = Trainer.from_params(params=params.pop('trainer'),
                              model=model,
                              iterator=iterator,
                              train_data=instances,
                              validation_data=None,
                              serialization_dir=serialization_dir)

# No need to see what the scores are before training,
# our trainer will show the loss over time.

# Train
trainer.train()

# See what the scores are after training
with torch.no_grad():
    tensor_dict = next(iterator(instances))
    tag_scores = model.forward(**tensor_dict)['tag_logits']

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is
    # DET NN V DET NN, the correct sequence!
    print(tag_scores)
    tag_ids = torch.argmax(tag_scores, dim=-1)[0].tolist()
    print([vocab.get_token_from_index(i, 'labels') for i in tag_ids])

shutil.rmtree(serialization_dir)
