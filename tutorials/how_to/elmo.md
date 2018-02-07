
Using pre-trained ELMo representations
--------------------------------------

Pre-trained contextual representations from large scale bidirectional
language models provide large improvements for nearly all supervised
NLP tasks.

This document describes how to add ELMo representations to your model using `allennlp`.
We also have a [tensorflow implementation](https://github.com/allenai/bilm-tf).

For more detail about ELMo, please see the publication ["Deep contextualized word representations"](https://openreview.net/forum?id=S1p31z-Ab).

## Using ELMo with existing `allennlp` models

In the simplest case, adding ELMo to an existing model is a simple
configuration change.  We provide a `TokenEmbedder` that accepts
character ids as input, runs the deep biLM and computes the ELMo representations
via a learned weighted combination.
Note that this simple case only includes one layer of ELMo representation
in the final model.
In some case (e.g. SQuAD and SNLI) we found that including multiple layers improved performance.  Multiple layers require code changes (see below).

We will use existing SRL model [configuration file](../../training_config/semantic_role_labeler.json) as an example to illustrate the changes.  Without ELMo, it uses 100 dimensional pre-trained GloVe vectors.

To add ELMo, there are three relevant changes.  First, modify the `text_field_embedder` section as follows:

```json
   "text_field_embedder": {
     "tokens": {
       "type": "embedding",
       "embedding_dim": 100,
       "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
       "trainable": true
     },
     "elmo":{
       "type": "elmo_token_embedder",
       "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
       "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
       "do_layer_norm": true,
       "dropout": 0.5
     }
```

Second, add a section to the `dataset_reader` to convert raw text to ELMo character id sequences in addition to GloVe ids:

```json
 "dataset_reader": {
   "type": "srl",
   "token_indexers": {
     "tokens": {
       "type": "single_id",
       "lowercase_tokens": true
     },
     "elmo": {
       "type": "elmo_characters"
     }
   }
 }
```

Third, modify the input dimension to the stacked LSTM encoder.
The baseline model uses a 200 dimensional input (100 dimensional GloVe embedding with 100 dimensional feature specifying the predicate location).
ELMo provides a 1024 dimension representation so the new dimension is 1224.

```json
    "stacked_encoder": {
      "type": "alternating_lstm",
      "input_size": 1224,
      "hidden_size": 300,
      "num_layers": 8,
      "recurrent_dropout_probability": 0.1,
      "use_highway": true
    },
```


## Using ELMo programmatically

If you need to include ELMo at multiple layers in a task model or you have other advanced use cases, you will need to create ELMo vectors
programatically.  This is easily done with the ElmoEmbedder class [(API doc)](https://github.com/allenai/allennlp/tree/master/allennlp/commands/elmo.py).


```python
from allennlp.commands.elmo import ElmoEmbedder

ee = ElmoEmbedder()

ee.embed_sentence("Bitcoin alone has a sixty percent share of global search.")

embeddings = ee.embed_sentence("Bitcoin alone has a sixty percent share of global search.".split())

# embeddings has shape (3, 11, 1024)
#   3    - the number of ELMo vectors.
#   11   - the number of words in the input sentence
#   1024 - the length of each ELMo vector
```

## Writing contextual representations to disk

You can write ELMo representations to disk with the `elmo` command.  The `elmo`
command will write all the biLM individual layer representations for a dataset
of sentences to an HDF5 file.  Here is an example of using the `elmo` command:

```bash
echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt
python -m allennlp.run elmo sentences.txt elmo_layers.hdf5
```

For more details, see `python -m allennlp.run elmo -h`.
