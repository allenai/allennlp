import sys
import torch
sys.path.append('./python')
import chunking

use_cuda = torch.cuda.is_available()

def main(argv):
  
    input_lang, output_lang, pairs, pairs_dev, max_length = chunking.data.initializeData()  
      
    hidden_size = 1029
    encoder1 = chunking.main.EncoderRNN(input_lang.n_words, hidden_size, input_lang, n_layers = 2)
    attn_decoder1 = chunking.main.AttnDecoderRNN(hidden_size, output_lang.n_words, max_length,
                               1, dropout_p=0.1)

    print("*** starting training ***")
    if use_cuda:
        print("*** using cuda to train ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    chunking.main.trainIters(encoder1, attn_decoder1, input_lang, output_lang, 150000, pairs, pairs_dev, max_length, print_every=1000)
    print("*** done training ***")
    print(chunking.main.validate(encoder1, attn_decoder1, input_lang, output_lang, pairs_dev, max_length, 4813))
    torch.save(encoder1, 'encoder.final.pt')
    torch.save(attn_decoder1, 'decoder.final.pt')
    

if __name__ == "__main__":
    main(sys.argv)
