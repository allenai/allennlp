# Using pre-trained ELMo representations

Pre-trained contextual representations from large scale bidirectional
language models provide large improvements for nearly all supervised
NLP tasks.

This document describes how to add ELMo representations to your model using `allennlp`.
We also have a [tensorflow implementation](https://github.com/allenai/bilm-tf).

For more detail about ELMo, please see the publication ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).

## Writing contextual representations to disk

You can write ELMo representations to disk with the `elmo` command.  The `elmo`
command will write all the biLM individual layer representations for a dataset
of sentences to an HDF5 file. The generated hdf5 file will contain line indices
of the original sentences as keys. Here is an example of using the `elmo` command:

```bash
echo "The cryptocurrency space is now figuring out to have the highest search on Google globally ." > sentences.txt
echo "Bitcoin alone has a sixty percent share of global search ." >> sentences.txt
allennlp elmo sentences.txt elmo_layers.hdf5 --all
```

If you'd like to use the ELMo embeddings without keeping the original dataset of
sentences around, using the `--include-sentence-indices` flag will write a
JSON-serialized string with a mapping from sentences to line indices to the
`"sentence_indices"` key.

For more details, see `allennlp elmo -h`. 

## Using ELMo programmatically

If you need to include ELMo at multiple layers in a task model or you have other advanced use cases, you will need to create ELMo vectors programatically.
This is easily done with the `Elmo` class [(API doc)](https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L27), which provides a mechanism to compute the weighted ELMo representations (Equation (1) in the paper).

This is a `torch.nn.Module` subclass that computes any number of ELMo
representations and introduces trainable scalar weights for each.
For example, this code snippet computes two layers of representations
(as in the SNLI and SQuAD models from our paper):

```python
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)

# embeddings['elmo_representations'] is length two list of tensors.
# Each element contains one layer of ELMo representations with shape
# (2, 3, 1024).
#   2    - the batch size
#   3    - the sequence length of the batch
#   1024 - the length of each ELMo vector
```

If you are not training a pytorch model, and just want numpy arrays as output
then use `allennlp.commands.elmo.ElmoEmbedder`.


## Using ELMo with existing `allennlp` models

In the simplest case, adding ELMo to an existing model is a simple
configuration change.  We provide a `TokenEmbedder` that accepts
character ids as input, runs the deep biLM and computes the ELMo representations
via a learned weighted combination.
Note that this simple case only includes one layer of ELMo representation
in the final model.
In some case (e.g. SQuAD and SNLI) we found that including multiple layers improved performance.  Multiple layers require code changes (see below).

We will use existing SRL model [configuration file](../../training_config/semantic_role_labeler.json) as an example to illustrate the changes.  Without ELMo, it uses 100 dimensional pre-trained GloVe vectors.

To add ELMo, there are three relevant changes.  First, modify the `text_field_embedder` section by adding an `elmo` section as follows:

```json
"text_field_embedder": {
  "tokens": {
    "type": "embedding",
    "embedding_dim": 100,
    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
    "trainable": true
  },
  "elmo": {
    "type": "elmo_token_embedder",
    "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
    "do_layer_norm": false,
    "dropout": 0.5
  }
}
```

Second, add an `elmo` section to the `dataset_reader` to convert raw text to ELMo character id sequences in addition to GloVe ids:

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

Third, modify the input dimension (`input_size`) to the stacked LSTM encoder.
The baseline model uses a 200 dimensional input (100 dimensional GloVe embedding with 100 dimensional feature specifying the predicate location).
ELMo provides a 1024 dimension representation so the new `input_size` is 1224.

```json
"encoder": {
  "type": "alternating_lstm",
  "input_size": 1224,
  "hidden_size": 300,
  "num_layers": 8,
  "recurrent_dropout_probability": 0.1,
  "use_highway": true
}
```

## Recommended hyper-parameter settings for `Elmo` class

When using ELMo, there are several hyper-parameters to set.  As a general rule, we have found
training to be relatively insensitive to the hyper-parameters, but nevertheless here are some
general guidelines for an initial training run.

* Include one layer of ELMo representations at the same location as pre-trained word representations.
* Set `do_layer_norm=False` when constructing the `Elmo` class.
* Add some dropout (0.5 is a good default value), either in the `Elmo` class directly, or in the next layer of your network.  If the next layer of the network includes dropout then set `dropout=0` when constructing the `Elmo` class.
* Add a small amount of L2 regularization to the scalar weighting parameters (`lambda=0.001` in the paper).  These are the parameters named `scalar_mix_L.scalar_parameters.X` where `X=[0, 1, 2]` indexes the biLM layer and `L` indexes the number of ELMo representations included in the downstream model.  Often performance is slightly higher for larger datasets without regularizing these parameters, but it can sometimes cause training to be unstable.

Finally, we have found that in some cases including pre-trained GloVe or other word vectors in addition to ELMo provides little to no improvement over just using ELMo and slows down training.  However, we recommend experimenting with your dataset and model architecture for best results.

## Notes on statefulness and non-determinism

The pre-trained biLM used to compute ELMo representations was trained without resetting the internal LSTM states between sentences.
Accordingly, the re-implementation in allennlp is stateful, and carries the LSTM states forward from batch to batch.
Since the biLM was trained on randomly shuffled sentences padded with special `<S>` and `</S>` tokens, it will reset the internal states to its own internal representation of sentence break when seeing these tokens.

There are a few practical implications of this:

* Due to the statefulness, the ELMo vectors are not deterministic and running the same batch multiple times will result in slightly different embeddings.
* After loading the pre-trained model, the first few batches will be negatively impacted until the biLM can reset its internal states.  You may want to run a few batches through the model to warm up the states before making predictions (although we have not worried about this issue in practice).
* It is important to always add the `<S>` and `</S>` tokens to each sentence.  The `allennlp` code handles this behind the scenes, but if you are handing padding and indexing in a different manner then take care to ensure this is handled appropriately.
