
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.common.file_utils import cached_path

options = cached_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json")
weights = cached_path("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")



# load the data
import json
with open('sentences_20_50_for_benchmark.json', 'r') as fin:
    sentences = json.load(fin)


import numpy

import torch
import time


for k in ['20', '50']:
    for use_vocab in [True, False]:
        sents = sentences[k][:64]

        vocab = list(set([word for sent in sents for word in sent]))
        print("vocab_size: ", len(vocab))
        if use_vocab:
            encoder = _ElmoBiLm(options, weights, vocab_to_cache=vocab)
        else:
            encoder = _ElmoBiLm(options, weights)
        encoder.eval()

        char_ids = batch_to_ids(sents)
        #sents = [sentences[k][0]]

        # warm up
        for i in range(5):
            _ = encoder(char_ids)

        # get the character encoder time
        t1 = time.time()
        for i in range(5):

            if use_vocab:
                character_ids_with_bos_eos, mask_with_bos_eos = encoder.add_bos_and_eos(char_ids)
                word_ids = encoder._get_word_ids_from_character_ids(character_ids_with_bos_eos)
                type_representation = encoder._word_embedding(word_ids)

            else:
                _ = encoder._token_embedder(char_ids)
        t2 = time.time()

        # now total time
        for i in range(5):
            _ = encoder(char_ids)
        t3 = time.time()

        if use_vocab:
            for i in range(5):
                encoder.create_cached_cnn_embeddings(vocab)
        t4 = time.time()

        total_time = (t3 - t2) / 5
        char_time = (t2 - t1) / 5
        vocab_creation_time = (t4 - t3) /5
        context_time = total_time - char_time
        print("Batch Size: ", k, "Cached Vocab: ",use_vocab)
        print(char_time * 1000, context_time * 1000, total_time * 1000)

        if use_vocab:
            print("Vocab creation time: ", vocab_creation_time * 1000)