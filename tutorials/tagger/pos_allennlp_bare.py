# from https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# pylint: disable=invalid-name,arguments-differ
from typing import Iterator, List

import torch
import torch.optim as optim

from allennlp.data import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

torch.manual_seed(1)

training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

class PosDatasetReader(DatasetReader):
    """
    Normally you'd read data from a file, but here we're just using a tiny in-memory dataset
    """
    def __init__(self) -> None:
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: List[str], tags: List[str]) -> Instance:
        tokens = [Token(word) for word in sentence]
        sentence_field = TextField(tokens, self.token_indexers)
        label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        return Instance(fields={"sentence": sentence_field,
                                "labels": label_field})


    def _read(self, _file_path: str) -> Iterator[Instance]:
        for sentence, tags in training_data:
            yield self.text_to_instance(sentence, tags)

reader = PosDatasetReader()
instances = reader.read(None)
vocab = Vocabulary.from_instances(instances)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 hidden2tag: FeedForward,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = hidden2tag

    def forward(self, sentence: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        embeddings = self.word_embeddings(sentence)
        mask = get_text_field_mask(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
hidden2tag = FeedForward(input_dim=HIDDEN_DIM,
                         num_layers=1,
                         hidden_dims=vocab.get_vocab_size('labels'),
                         activations=lambda x: x)

model = LstmTagger(word_embeddings, lstm, hidden2tag, vocab)

optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BasicIterator(batch_size=2)
iterator.index_with(vocab)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    for tensor_dict in iterator(instances, num_epochs=1):
        output = model.forward(**tensor_dict)
        print(output['tag_logits'])


for epoch in range(500):  # again, normally you would NOT do 300 epochs, it is toy data
    for tensor_dict in iterator(instances, num_epochs=1):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run the forward pass
        outputs = model(**tensor_dict)

        # Step 3. Compute the gradients and optimizer step
        outputs['loss'].backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    for tensor_dict in iterator(instances, num_epochs=1):
        output = model.forward(**tensor_dict)

        # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
        # for word i. The predicted tag is the maximum scoring tag.
        # Here, we can see the predicted sequence below is 0 1 2 0 1
        # since 0 is index of the maximum value of row 1,
        # 1 is the index of maximum value of row 2, etc.
        # Which is DET NOUN VERB DET NOUN, the correct sequence!
        tag_scores = output['tag_logits']
        print(tag_scores)
        print(torch.argmax(tag_scores, dim=-1))
