import sys
import torch
import h5py
import codecs
from chunking.main import prepareData
from chunking.main import NeuralChunker
import chunking

__all__ = ['initializeData', 'initializeChunker']

def initializeData():
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
    wordVecs = chunking.main.WordVectors(special_tokens, elmo_vecs)    
    input_lang, output_lang, pairs, MAX_LENGTH = prepareData('esrc', 'etgt', wordVecs)
    input_lang_dev, output_lang_dev, pairs_dev, MAX_LENGTH_DEV = prepareData('esrcd', 'etgtd', wordVecs)
    input_lang = wordVecs
    input_lang_dev = wordVecs
    return input_lang, output_lang, pairs, pairs_dev, MAX_LENGTH    

def initializeChunker(encoderFile, decoderFile):
    input_lang, output_lang, pairs, pairs_dev, max_length = initializeData() 
    chunker = NeuralChunker(encoderFile, decoderFile, input_lang, output_lang, max_length)
    return chunker, pairs, pairs_dev

