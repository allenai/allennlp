

def main(argv):
    
    hidden_size = 1029
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                               1, dropout_p=0.1)

    if use_cuda:
        print("*** using cuda to train ***")
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(encoder1, attn_decoder1, 150000, print_every=1000)
    print("*** done training ***")
    print(validate(encoder1, attn_decoder1, pairs_dev, 4813))
    torch.save(encoder1, 'encoder.final.pt')
    torch.save(attn_decoder1, 'decoder.final.pt')
 

if __name__ == "__main__":
    main(sys.argv)
