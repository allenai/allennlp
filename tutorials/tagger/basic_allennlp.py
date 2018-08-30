"""
This is the AllenNLP equivalent of
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#example-an-lstm-for-part-of-speech-tagging
with the following changes:

 1. read data from files
 2. separate test data and validation data
 3. add tqdm with loss metrics
 4. early stopping based on validation loss
 5. track accuracy during training / validation
"""
# pylint: disable=invalid-name,arguments-differ,redefined-outer-name
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer

torch.manual_seed(1)

class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)

reader = PosDatasetReader()
train_dataset = reader.read('tutorials/tagger/training.txt')
validation_dataset = reader.read('tutorials/tagger/validation.txt')
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

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
        self.accuracy = CategoricalAccuracy()

    def forward(self, sentence: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.word_embeddings(sentence)
        mask = get_text_field_mask(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = LstmTagger(word_embeddings, lstm, vocab)
optimizer = optim.SGD(model.parameters(), lr=0.1)

iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000)

# No need to see what the scores are before training,
# our trainer will show the loss over time.

# Train
trainer.train()

# See what the scores are after training
# Make predictions
predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_scores = predictor.predict("The dog ate the apple")['tag_logits']
print(tag_scores)
tag_ids = np.argmax(tag_scores, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
