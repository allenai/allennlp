import sys
import torch
sys.path.append('./python')

from chunking.data import initializeData
from chunking.eval import validate
from chunking.train import trainItersElmo, EncoderRNNElmo, AttnDecoderRNN

use_cuda = torch.cuda.is_available()

def main(argv):
 
    print("*** initializing data ***")
    output_lang, pairs, pairs_dev, max_length = initializeData('data/wsj.unlabeled.elmo.train.txt', 'data/wsj.unlabeled.elmo.dev.txt', max_input_length = 30)  
      
    hidden_size = 1029
    if use_cuda:
        device = 0
    else:
        device = -1    
    encoder1 = EncoderRNNElmo(hidden_size, device=device, n_layers = 2)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length,
                               1, dropout_p=0.1)

    print("*** starting training ***")
    if use_cuda:
        print("*** using cuda to train ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainItersElmo(encoder1, attn_decoder1, output_lang, 150000, pairs, pairs_dev, max_length, print_every=100)
    # experimental
    #    trainItersElmo(encoder1, attn_decoder1, output_lang, 750, 200, pairs, pairs_dev, max_length, print_every=1, save_every=10)
    print("*** done training ***")
    print(validate(encoder1, attn_decoder1, output_lang, pairs_dev, max_length, 4813))
    torch.save(encoder1, 'encoder.final.pt')
    torch.save(attn_decoder1, 'decoder.final.pt')
    

if __name__ == "__main__":
    main(sys.argv)
