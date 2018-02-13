import sys
import torch
sys.path.append('./python')

from chunking.data import initializeData, initializeChunker
from chunking.eval import validate
from chunking.train import trainItersElmo, EncoderRNNElmo, AttnDecoderRNN

use_cuda = torch.cuda.is_available()

def main(argv):
 
    print("*** initializing data ***")
    output_lang, pairs, pairs_dev, max_length = initializeData('data/mturk.005.train.txt', 'data/mturk.005.dev.txt', max_input_length = 120)

      
    hidden_size = 1034
    if use_cuda:
        device = 0
    else:
        device = -1
    
        
    chunker, pairs, pairs_dev = initializeChunker(
        'results/run_008/encoder.final.pt',
        'results/run_008/decoder.final.pt',
        'data/mturk.005.train.txt',
        'data/mturk.005.dev.txt',
        max_input_length = 120
    )
        
    encoder1 = chunker.encoder
    attn_decoder1 = chunker.decoder

    if use_cuda:
        print("*** using cuda to train ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()


    print("*** Analyzing ***")
    print(validate(encoder1, attn_decoder1, output_lang, pairs_dev, max_length, 100))
    

if __name__ == "__main__":
    main(sys.argv)
