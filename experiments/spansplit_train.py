import sys
import torch
import h5py
import codecs
sys.path.append('./python')
from chunking.main import prepareData
import chunking

use_cuda = torch.cuda.is_available()

def main(argv):
    
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
    
    hidden_size = 1029
    encoder1 = chunking.main.EncoderRNN(input_lang.n_words, hidden_size, wordVecs, n_layers=2)
    attn_decoder1 = chunking.main.AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH,
                               1, dropout_p=0.1)

    print("*** starting training ***")
    if use_cuda:
        print("*** using cuda to train ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    chunking.main.trainIters(encoder1, attn_decoder1, input_lang, output_lang, 300000, pairs, pairs_dev, MAX_LENGTH, print_every=1000)
    print("*** done training ***")
    print(chunking.main.validate(encoder1, attn_decoder1, input_lang, output_lang, pairs_dev, MAX_LENGTH, 4813))
    torch.save(encoder1, 'encoder.final.pt')
    torch.save(attn_decoder1, 'decoder.final.pt')
    

if __name__ == "__main__":
    main(sys.argv)
