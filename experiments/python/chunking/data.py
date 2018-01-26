import sys
import torch
import h5py
import codecs
import numpy as np
from torch.autograd import Variable


__all__ = ['initializeData', 'initializeChunker', 'prepareData', 'WordVectors', 'variableFromSentence', 'variablesFromPair']

EOS_token = 1
use_cuda = torch.cuda.is_available()


class WordVectors:
    def __init__(self, special_tokens, elmo_vectors):
        self.special_tokens = special_tokens
        self.special_token_to_id = {tok: i for (i, tok) in enumerate(special_tokens)}
        self.dimension = len(special_tokens) + 1024
        self.elmo_vectors = {k: v for (k, v) in elmo_vectors}
        self.words = self.special_tokens + [tok for (tok, v) in elmo_vectors]
        self.index2word = {k: v for (k, v) in enumerate(self.words)}
        self.word2index = {v: k for (k, v) in enumerate(self.words)}
        self.n_words = len(self.words)

    def dumpAsTensor(self):
        veclist = [self.getWordVector(self.index2word[i]) for i in range(self.n_words)]
        return np.concatenate(veclist, axis=0).reshape(-1, self.dimension)
    
    def getWordVector(self, word):
        if word in self.special_token_to_id:
            return np.eye(self.dimension)[self.special_token_to_id[word]]
        else:
            return np.concatenate([np.zeros(len(self.special_tokens)), self.elmo_vectors[word]], axis=0)
    
class TargetLang:
    def __init__(self, wordVectors):
        self.word2index = {v: k for (k, v) in enumerate(wordVectors.special_tokens)}
        self.index2word = {k: v for (k, v) in enumerate(wordVectors.special_tokens)}
        self.n_words = len(wordVectors.special_tokens)
        self.wordVectors = wordVectors
        self.dimension = wordVectors.dimension

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def prepareData(train_file, wordVecs):
    print("Reading lines...")
    # Read the file and split into lines
    lines = codecs.open(train_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.split('\t')] for l in lines]
    output_lang = TargetLang(wordVecs)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    max_encoding_length = 1
    for pair in pairs:
        if len(pair[0].split(' ')) > max_encoding_length:
            max_encoding_length = len(pair[0].split(' '))
        output_lang.addSentence(pair[1])
    print("Max encoding length: %s" % max_encoding_length)
    print("Counted target words: {}".format(output_lang.n_words))
    print(output_lang.index2word)
    return output_lang, pairs, max_encoding_length + 5


# Maps the sentence tokens into their ids.
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# Maps the sentence into a PyTorch vector (containing the token ids).
def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

# Maps the source and target sentence into PyTorch vectors (of the token ids).
def variablesFromPair(pair, input_lang, output_lang):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)

def initializeData(train_file = 'data/esrc-etgt.txt', dev_file = 'data/esrcd-etgtd.txt'):
    print("*** loading elmo vectors ***")
    elmo_vecs = []
    with h5py.File('data/question_bank_elmo_embeddings.hdf5', 'r') as open_file:
        with codecs.open('data/qbank.tokens.txt', encoding='utf-8') as qbank_sents:
            for (sent_id, sent) in enumerate(qbank_sents):
                sentence_embedding = open_file[str(sent_id)][...]
                a = torch.from_numpy(sentence_embedding[0])
                b = torch.from_numpy(sentence_embedding[1])
                c = torch.from_numpy(sentence_embedding[2])
                result = (a + b + c) / 3.0
                sent_toks = sent.strip().split()
                for (tok_id, tok) in enumerate(sent_toks):
                    canonical_tok = '{}__{}__{}'.format(tok_id, sent_id, tok)
                    elmo_vecs.append((canonical_tok, result[tok_id]))
   
    print("*** compiling vocab ***")
    special_tokens = ['sos', 'eos', '[[[', ']]]', '<unk>']     
    wordVecs = WordVectors(special_tokens, elmo_vecs)    
    output_lang, pairs, MAX_LENGTH = prepareData(train_file, wordVecs)
    output_lang_dev, pairs_dev, MAX_LENGTH_DEV = prepareData(dev_file, wordVecs)
    input_lang = wordVecs
    input_lang_dev = wordVecs
    return input_lang, output_lang, pairs, pairs_dev, MAX_LENGTH    

def initializeChunker(encoderFile, decoderFile):
    input_lang, output_lang, pairs, pairs_dev, max_length = initializeData() 
    chunker = NeuralChunker(encoderFile, decoderFile, input_lang, output_lang, max_length)
    return chunker, pairs, pairs_dev


