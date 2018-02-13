import sys
import torch
sys.path.append('./python')

from chunking.data import initializeData, initializeChunker, NeuralChunker
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
            
    chunker = NeuralChunker('encoder.final.pt','decoder.final.pt', output_lang, max_length)

    encoder1 = chunker.encoder
    attn_decoder1 = chunker.decoder
    
    print("*** starting evaluation ***")
    if use_cuda:
        print("*** using cuda to eval ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainItersElmo(encoder1, attn_decoder1, output_lang, 0, pairs, pairs_dev, max_length, print_every=100)
    # experimental
    #    trainItersElmo(encoder1, attn_decoder1, output_lang, 750, 200, pairs, pairs_dev, max_length, print_every=1, save_every=10)
    print(validate(torch.load('encoder.final.pt'), torch.load('decoder.final.pt'), output_lang, pairs_dev, max_length, 100))
#    torch.save(encoder1, 'encoder.final.pt')
#    torch.save(attn_decoder1, 'decoder.final.pt')


if __name__ == "__main__":
    main(sys.argv)
