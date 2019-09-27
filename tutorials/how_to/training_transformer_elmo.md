# Training Transformer ELMo

This document describes how to train and use a transformer-based version of ELMo with `allennlp`. The model is a port of the the one described in [Dissecting Contextual Word Embeddings: Architecture and Representation](https://www.semanticscholar.org/paper/4dc99343fdc57cf974746e9549c6ee56f013cee5) by Peters et al. You can find a pretrained version of this model [here](https://allennlp.s3.amazonaws.com/models/transformer-elmo-2019.01.10.tar.gz).

## Training

1. Obtain training data from https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz.
    ```
    export BIDIRECTIONAL_LM_DATA_PATH=$PWD'/1-billion-word-language-modeling-benchmark-r13output'
    export BIDIRECTIONAL_LM_TRAIN_PATH=$BIDIRECTIONAL_LM_DATA_PATH'/training-monolingual.tokenized.shuffled/*'
    ```
2. Obtain vocab.
    ```
    pip install --user awscli
    mkdir vocabulary
    export BIDIRECTIONAL_LM_VOCAB_PATH=$PWD'/vocabulary'
    cd $BIDIRECTIONAL_LM_VOCAB_PATH
    aws --no-sign-request s3 cp s3://allennlp/models/elmo/vocab-2016-09-10.txt .
    cat vocab-2016-09-10.txt | sed 's/<UNK>/@@UNKNOWN@@/' > tokens.txt
    # Avoid creating garbage namespace.
    rm vocab-2016-09-10.txt
    echo '*labels\n*tags' > non_padded_namespaces.txt
    ```
3. Run training. Note: `training_config` refers to [this directory](../../training_config).
    ```
    # The multiprocess dataset reader and iterator use many file descriptors,
    # so we increase the relevant ulimit here to help.
    # See https://pytorch.org/docs/stable/multiprocessing.html#file-descriptor-file-descriptor
    # for a description of the underlying issue.
    ulimit -n 4096
    # Location of repo for training_config.
    cd allennlp
    allennlp train training_config/bidirectional_language_model.jsonnet --serialization-dir output_path
    ```
4. Wait. This will take days. (Example results here are from a model trained for just 4 epochs.)
5. Evaluate. There is one gotcha here, which is that we discard 3 sentences for being too long (otherwise we'd exhaust GPU memory). If we wanted to report this number formally (in a paper or similar), we'd need to handle this differently.
    ```
    allennlp evaluate --cuda-device 0 -o '{"iterator": {"base_iterator": {"maximum_samples_per_batch": ["num_tokens", 500] }}}' output_path/model.tar.gz $BIDIRECTIONAL_LM_DATA_PATH/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100
    ```

    A model trained for 4 epochs gives:
    ```
    2018-12-12 05:42:53,711 - INFO - allennlp.commands.evaluate - loss: 3.745238332322373

    ipython
    In [1]: import math; math.exp(3.745238332322373) # To compute perplexity
    Out[1]: 42.3190920245054
    ```

## Using Transformer ELMo with existing `allennlp` models

Using Transformer ELMo is essentially the same as using regular ELMo. See [this documentation](elmo.md#using-elmo-with-existing-allennlp-models) for details on how to do that.

The one exception is that inside the `text_field_embedder` block in your training config you should replace

```json
"text_field_embedder": {
  "token_embedders": {
    "elmo": {
      "type": "elmo_token_embedder",
      "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.5
    }
  }
},
```

with

```json
"text_field_embedder": {
  "token_embedders": {
    "elmo": {
      "type": "bidirectional_lm_token_embedder",
      "archive_file": std.extVar('BIDIRECTIONAL_LM_ARCHIVE_PATH'),
      "dropout": 0.2,
      "bos_eos_tokens": ["<S>", "</S>"],
      "remove_bos_eos": true,
      "requires_grad": false
    }
  }
},
```
.

For an example of this see the config for a [Transformer ELMo augmented constituency parser](../../training_config/constituency_parser_transformer_elmo.jsonnet) and compare with the [original ELMo augmented constituency parser](../../training_config/constituency_parser_elmo.jsonnet).

## Calling the `BidirectionalLanguageModelTokenEmbedder` directly

Of course, you can also directly call the embedder in your programs:

```
from allennlp.modules.token_embedders.bidirectional_language_model_token_embedder import BidirectionalLanguageModelTokenEmbedder
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers.token import Token
import torch

lm_model_file = "output_path/model.tar.gz"

sentence = "It is raining in Seattle ."
tokens = [Token(word) for word in sentence.split()]

lm_embedder = BidirectionalLanguageModelTokenEmbedder(
    archive_file=lm_model_file,
    dropout=0.2,
    bos_eos_tokens=["<S>", "</S>"],
    remove_bos_eos=True,
    requires_grad=False
)

indexer = ELMoTokenCharactersIndexer()
vocab = lm_embedder._lm.vocab
character_indices = indexer.tokens_to_indices(tokens, vocab, "elmo")["elmo"]

# Batch of size 1
indices_tensor = torch.LongTensor([character_indices])

# Embed and extract the single element from the batch.
embeddings = lm_embedder(indices_tensor)[0]

for word_embedding in embeddings:
    print(word_embedding)
```

Note: This sidesteps our data loading and batching mechanisms for brevity. See [our main tutorial](https://allennlp.org/tutorials) for an exposition of how they function.
