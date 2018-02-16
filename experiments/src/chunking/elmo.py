import torch
from torch.autograd import Variable
from torch.nn import Module
import h5py

import numpy as np

from allennlp.data.dataset import Dataset
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.nn.util import remove_sentence_boundaries
from allennlp.modules.elmo import _ElmoBiLm

from src.chunking.data import variableFromSentence

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

use_cuda = torch.cuda.is_available()

if use_cuda:
    elmo_bilm = _ElmoBiLm(options_file, weight_file).cuda()
else:
    elmo_bilm = _ElmoBiLm(options_file, weight_file)


indexer = ELMoTokenCharactersIndexer()

__all__ = ['elmo_bilm', 'embed_sentence', 'ElmoEmbedder', 'variablesFromPairElmo', 'elmo_variable_from_sentence']


class ElmoEmbedder(Module):
    def __init__(self, elmo_bilm, special_tokens, device):
        super(ElmoEmbedder, self).__init__()
        self.elmo_bilm = elmo_bilm
        self.device = device
        self.special_tokens = special_tokens
        self.elmo_id_to_special_token = {tuple(token_to_elmo_id(tok).view(-1).data): tok_id for (tok_id, tok) in enumerate(special_tokens)}
        self.special_token_to_id = {tok: i for (i, tok) in enumerate(special_tokens)}
        self.dimension = len(special_tokens) + 1024
        self.cached_embedding = (-1, None)
        
    def forward(self, input, token_index):
        input_code = tuple([tuple(input[0][i].data) for i in range(input.shape[1])])
        token_code = tuple(input[0][token_index].data)
        if token_code in self.elmo_id_to_special_token:
            tok_id = self.elmo_id_to_special_token[token_code]
            result = Variable(torch.from_numpy(np.eye(self.dimension)[tok_id]).float())
        elif input_code == self.cached_embedding[0]:
            embedded = self.cached_embedding[1]
            result = torch.cat([Variable(torch.from_numpy(np.zeros(len(self.special_tokens))).float()), 
                                Variable(embedded[token_index])])            
        else:
            embedded = embed_numerical_sent(input, elmo_bilm, self.device) 
            self.cached_embedding = (input_code, embedded)
            result = torch.cat([Variable(torch.from_numpy(np.zeros(len(self.special_tokens))).float()), 
                                Variable(embedded[token_index])])
        if use_cuda:
            return result.cuda()
        else:
            return result
        
        
def character_ids_to_embeddings(character_ids, elmo_bilm, device):
    # returns (batch_size, 3, num_times, 1024) embeddings and (batch_size, num_times) mask
    if device >= 0:
        character_ids = character_ids.cuda(device=device)
    bilm_output = elmo_bilm(character_ids)
    layer_activations = bilm_output['activations']
    mask_with_bos_eos = bilm_output['mask']
    without_bos_eos = [remove_sentence_boundaries(layer, mask_with_bos_eos)
            for layer in layer_activations]
    # without_bos_eos is a 3 element list of (batch_size, num_times, dim) arrays
    activations = torch.cat([ele[0].unsqueeze(1) for ele in without_bos_eos], dim=1)
    mask = without_bos_eos[0][1]
    return activations, mask

    
def embed_numerical_sent(input_var, elmo_bilm, device):
    embeddings, mask = character_ids_to_embeddings(input_var, elmo_bilm, device)
    sent_embeddings = []
    for i in range(1):
        length = int(mask[i, :].sum())
        sentence_embedding = embeddings[i, :, :length, :].data.cpu().numpy()
        a = torch.from_numpy(sentence_embedding[0])
        b = torch.from_numpy(sentence_embedding[1])
        c = torch.from_numpy(sentence_embedding[2])
        result = (a + b + c) / 3.0
        sent_embeddings.append(result)
    return sent_embeddings[0]

def token_to_elmo_id(token):
    tokens = [Token(token)]
    field = TextField(tokens, {'character_ids': indexer})
    instance = Instance({"elmo": field})
    instances = [instance]
    dataset = Dataset(instances)
    vocab = Vocabulary()
    for instance in dataset.instances:
        instance.index_fields(vocab)
    #dataset.index_instances(vocab) # replaced by above, so that there's no progress bar
    return dataset.as_tensor_dict()['elmo']['character_ids']


def batch_to_ids(batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Dataset(instances)
    vocab = Vocabulary()
    for instance in dataset.instances:
        instance.index_fields(vocab)
    #dataset.index_instances(vocab)        #replaced by above, so there's no progress bar
    return dataset.as_tensor_dict()['elmo']['character_ids']

def elmo_variable_from_sentence(sent):
    tokens = sent.split()
    tokens.append('eos')
    return batch_to_ids([tokens])

# Maps the source and target sentence into PyTorch vectors (of the token ids).
def variablesFromPairElmo(pair, output_lang):
    input_variable = elmo_variable_from_sentence(pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def pad(tokens, desired_length, padder):
    padding = [padder] * (desired_length - len(tokens))
    return tokens + padding

def elmo_variable_from_sentences(sents):
    tokens = [sent.split() for sent in sents]
    max_size = max([len(x) for x in tokens]) + 1
    padded = [pad(tok, max_size, padder='eos') for tok in tokens]
    return batch_to_ids(padded)
