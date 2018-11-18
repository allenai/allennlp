"""
This is mostly just the tutorial from
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
with the following changes:

 1. read data from files
 2. separate test data and validation data
 3. add tqdm with loss metrics
 4. early stopping based on validation loss
 5. track accuracy during training / validation
"""
# pylint: disable=invalid-name,redefined-outer-name
from typing import Iterable, Mapping, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

torch.manual_seed(1)

def load_data(file_path: str) -> Tuple[List[str], List[str]]:
    """
    One sentence per line, formatted like

        The###DET dog###NN ate###V the###DET apple###NN

    Returns a list of pairs (tokenized_sentence, tags)
    """
    data = []

    with open(file_path) as f:
        for line in f:
            pairs = line.strip().split()
            sentence, tags = zip(*(pair.split("###") for pair in pairs))
            data.append((sentence, tags))

    return data

training_data = load_data('tutorials/tagger/training.txt')
validation_data = load_data('tutorials/tagger/validation.txt')

def prepare_sequence(seq: Iterable[str], to_ix: Mapping[str, int]) -> torch.Tensor:
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

word_to_ix: Dict[str, int] = {}
tag_to_ix: Dict[str, int] = {}

for sent, tags in training_data + validation_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int) -> None:
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

validation_losses = []
patience = 10

for epoch in range(1000):
    training_loss = 0.0
    validation_loss = 0.0

    for dataset, training in [(training_data, True), (validation_data, False)]:
        correct = total = 0
        torch.set_grad_enabled(training)
        t = tqdm.tqdm(dataset)
        for i, (sentence, tags) in enumerate(t):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)

            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)

            predictions = tag_scores.max(-1)[1]
            correct += (predictions == targets).sum().item()
            total += len(targets)
            accuracy = correct / total

            if training:
                loss.backward()
                training_loss += loss.item()
                t.set_postfix(training_loss=training_loss / (i + 1), accuracy=accuracy)
                optimizer.step()
            else:
                validation_loss += loss.item()
                t.set_postfix(validation_loss=validation_loss / (i + 1), accuracy=accuracy)

    validation_losses.append(validation_loss)
    if (patience and
                len(validation_losses) >= patience and
                validation_losses[-patience] == min(validation_losses[-patience:])):
        print("patience reached, stopping early")
        break

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is
    # DET NN V DET NN, the correct sequence!
    print(tag_scores)
    tag_ids = torch.argmax(tag_scores, dim=-1).tolist()
    print([["DET", "NN", "V"][i] for i in tag_ids])
