import sys
import torch
import h5py
import codecs
import numpy as np
from torch.autograd import Variable

__all__ = ['initializeData', 'initializeChunker', 'prepareData', 'WordVectors', 'variableFromSentence', 'variablesFromPair']

EOS_token = 1
use_cuda = torch.cuda.is_available()
    
class NeuralChunker:
    def __init__(self, encoder_file, decoder_file, output_lang, max_length):
        self.encoder = torch.load(encoder_file)
        self.decoder = torch.load(decoder_file)
        self.output_lang = output_lang
        self.max_length = max_length
        
    def chunk(self, input_sent):
        return evaluate(self.encoder, self.decoder, self.output_lang, input_sent, self.max_length)[0]

    
class TargetLang:
    def __init__(self, special_tokens):
        self.word2index = {v: k for (k, v) in enumerate(special_tokens)}
        self.index2word = {k: v for (k, v) in enumerate(special_tokens)}
        self.n_words = len(special_tokens)

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

def prepareData(train_file, special_tokens, max_length = -1):
    print("Reading lines...")
    # Read the file and split into lines
    lines = codecs.open(train_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[s.strip() for s in l.split('\t')] for l in lines]
    output_lang = TargetLang(special_tokens)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    max_encoding_length = 1
    output_pairs = []
    for pair in pairs:
        input_toks = pair[0].split(' ')
        if len(input_toks) > max_encoding_length:
            max_encoding_length = len(input_toks)
        if max_length < 0 or len(input_toks) < max_length:
            output_pairs.append(pair)
            output_lang.addSentence(pair[1])
    if max_encoding_length > 0:
        max_encoding_length = max_length + 1
    else:
        max_encoding_length += 1
    print("Max encoding length: %s" % max_encoding_length)
    print("Filtered down to %s sentence pairs" % len(output_pairs))
    print("Counted target words: {}".format(output_lang.n_words))
    print(output_lang.index2word)
    return output_lang, output_pairs, max_encoding_length


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

def initializeData(train_file, dev_file, max_input_length = -1):   
    print("*** compiling target vocab ***")
    special_tokens = ['sos', 'eos', '<?>', '<unk>', '_a', '_b', '_c', '_d', '_e', '_f']
    output_lang, pairs, MAX_LENGTH = prepareData(train_file, special_tokens, max_input_length)
    output_lang_dev, pairs_dev, MAX_LENGTH_DEV = prepareData(dev_file, special_tokens, max_input_length)
    max_length = max(MAX_LENGTH, MAX_LENGTH_DEV)
    return output_lang, pairs, pairs_dev, max_length

def pad(tokens, desired_length, padder):
    padding = [padder] * (desired_length - len(tokens))
    return tokens + padding

# Maps the sentence into a PyTorch vector (containing the token ids).
def target_variable_from_sentences(lang, sentences):
    indexes = [indexesFromSentence(lang, sent) for sent in sentences]
    max_size = max([len(x) for x in indexes]) + 1
    padded = [torch.LongTensor(pad(index, max_size, padder=EOS_token)) for index in indexes]
    
    result = Variable(torch.stack(padded))
    if use_cuda:
        return result.cuda()
    else:
        return result
    
def initializeChunker(encoderFile, decoderFile, train_file, dev_file, max_input_length):
    output_lang, pairs, pairs_dev, max_length = initializeData(train_file, dev_file, max_input_length = max_input_length)  
    chunker = NeuralChunker(encoderFile, decoderFile, output_lang, max_length)
    return chunker, pairs, pairs_dev


