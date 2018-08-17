# pylint: disable=invalid-name,arguments-differ
#
# This is a slavish attempt to mimic the bare PyTorch version;
# in particular, it's not idiomatic AllenNLP at all.
from typing import List

import torch
import torch.optim as optim

from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

torch.manual_seed(1)

training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

def make_instance(sentence: List[str], tags: List[str]) -> Instance:
    tokens = [Token(token) for token in sentence]
    sentence_field = TextField(tokens, {"tokens": SingleIdTokenIndexer()})
    label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
    return Instance(fields={"sentence": sentence_field,
                            "labels": label_field})

instances = [make_instance(sentence, tags) for sentence, tags in training_data]
vocab = Vocabulary.from_instances(instances)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LstmTagger(Model):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = BasicTextFieldEmbedder({
                "tokens": Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=embedding_dim)
        })

        self.encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True))
        self.hidden2tag = FeedForward(input_dim=hidden_dim,
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


model = LstmTagger(EMBEDDING_DIM, HIDDEN_DIM, vocab)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Index the instances using our vocabulary.
# If you used our trainer this would happen automatically.
for instance in instances:
    instance.index_fields(vocab)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    tensor_dict = Batch(instances[:1]).as_tensor_dict()
    output = model.forward(**tensor_dict)
    print(output['tag_logits'])


for epoch in range(500):  # again, normally you would NOT do 300 epochs, it is toy data
    for instance in instances:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network.
        # If you used our trainer this would happen automatically.
        tensor_dict = Batch([instance]).as_tensor_dict()

        # Step 3. Run the forward pass
        outputs = model(**tensor_dict)

        # Step 4. Compute the gradients and optimizer step
        outputs['loss'].backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    tensor_dict = Batch(instances[:1]).as_tensor_dict()
    tag_scores = model.forward(**tensor_dict)['tag_logits']

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is
    # DET NN V DET NN, the correct sequence!
    print(tag_scores)
    tag_ids = torch.argmax(tag_scores, dim=-1)[0].tolist()
    print([vocab.get_token_from_index(i, 'labels') for i in tag_ids])
