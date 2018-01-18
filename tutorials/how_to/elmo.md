
Using pre-trained ELMo representations
--------------------------------------

Pre-trained contextual representations from large scale bidirectional
language models provide large improvements for nearly all supervised
NLP tasks.

This document describes how to add ELMo representations to your model using `allennlp`.
We also have a tensorflow implementation [here](https://github.com/allenai/bilm-tf).

Reference: ["Deep contextualized word representations"](https://openreview.net/forum?id=S1p31z-Ab)


## Installing

After installing `allennlp`, download the pre-trained options and weight files.

* [options file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [weight file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)


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
       "options_file": "/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json",
       "weight_file": "/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
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


## Usage template

Use the `Elmo` class directly [(API doc)](https://allenai.github.io/allennlp-docs/api/allennlp.modules.elmo.html)
to include ELMo at multiple layers in a task model, or for more advanced uses.


```python
# Compute multiple layers of ELMo representations from raw text

from allennlp.modules.elmo import Elmo
from allennlp.data.dataset import Dataset
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer


options_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json'
weight_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
use_gpu = False


# a helper function
indexer = ELMoTokenCharactersIndexer()
def batch_to_ids(batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Dataset(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']


# Create the ELMo class.  This example computes two output representation
# layers each with separate layer weights.
# We recommend adding dropout (50% is good default) either here or elsewhere
# where ELMo is used (e.g. in the next layer bi-LSTM).
elmo = Elmo(options_file, weight_file, num_output_representations=2,
            do_layer_norm=False, dropout=0)

if use_gpu:
    elmo.cuda()

# Finally, compute representations.
# The input is tokenized text, without any normalization.
batch = [
    'Pre-trained biLMs compute representations useful for NLP tasks .'.split(),
    'They give state of the art performance for many tasks .'.split(),
    'A third sentence .'.split()
]

# character ids is size (3, 11, 50)
character_ids = batch_to_ids(batch)
if use_gpu:
    character_ids = character_ids.cuda()

representations = elmo(character_ids)
# representations['elmo_representations'] is a list with two elements,
#   each is a tensor of size (3, 11, 1024).  Sequences shorter then the
#   maximum sequence are padded on the right, with undefined value where padded.
# representations['mask'] is a (3, 11) shaped sequence mask.
```

## Writing contextual representations to disk

See [write_elmo_representations_to_file.py](../../scripts/write_elmo_representations_to_file.py) for a script to dump all of the biLM individual layer representations for a dataset to hdf5 file.

