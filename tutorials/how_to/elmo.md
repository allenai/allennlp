# ELMo: Deep contextualized word representations

Pre-trained contextual representations of words from large scale bidirectional
language models provide large improvements over GloVe/word2vec baselines
for many supervised NLP tasks including question answering, coreference,
semantic role labeling, classification, and syntactic parsing.

This document describes how to add ELMo representations to your model using pytorch and `allennlp`.
We also have a [tensorflow implementation](https://github.com/allenai/bilm-tf).

For more detail about ELMo, please see the publication ["Deep contextualized word representations", NAACL 2018](https://www.aclweb.org/anthology/N18-1202) or the [ELMo section of the AllenNLP website](https://allennlp.org/elmo).

Citations:

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```

```
@inproceedings{Gardner2017AllenNLP,
  title={{AllenNLP}: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2018},
  booktitle={ACL workshop for NLP Open Source Software}
}
```

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
`"sentence_indices"` key.  For more details on command-line arguments, see 
`allennlp elmo -h`. 

Once you've written out ELMo vectors to HDF5, you can read them with various HDF5
libraries, such as h5py:

```
> import h5py
> h5py_file = h5py.File("elmo_layers.hdf5", 'r')
> embedding = h5py_file.get("0")
> assert(len(embedding) == 3) # one layer for each vector
> assert(len(embedding[0]) == 16) # one entry for each word in the source sentence
```

## Using ELMo as a PyTorch `Module` to train a new model

To train a model using ELMo, use the allennlp.modules.elmo.Elmo class ([API doc](https://github.com/allenai/allennlp/blob/master/allennlp/modules/elmo.py#L27)). This class provides a mechanism to compute the weighted ELMo representations (Equation (1) in the paper) as a PyTorch tensor.  The weighted average can be learned as part of a larger model and typically works best for using ELMo to improving performance on a particular task.

This is a `torch.nn.Module` subclass that computes any number of ELMo
representations and introduces trainable scalar weights for each.
For example, this code snippet computes two layers of representations
(as in the SNLI and SQuAD models from our paper):

```python
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
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

## Using ELMo interactively

You can use ELMo interactively (or programatically) with iPython.  The `allennlp.commands.elmo.ElmoEmbedder` class provides the easiest way to process one or many sentences with ELMo, but it returns numpy arrays so it is meant for use as a standalone command and not within a larger model.  For example, if you would like to learn a weighted average of the ELMo vectors then you need to use `allennlp.modules.elmo.Elmo` instead.

The ElmoEmbedder class returns three vectors for each word, each vector corresponding to a layer in the ELMo LSTM output. The first layer corresponds to the context insensitive token representation, followed by the two LSTM layers. See the ELMo paper or follow up work at EMNLP 2018 for a description of what types of information is captured in each layer.

```
$ ipython
> from allennlp.commands.elmo import ElmoEmbedder
> elmo = ElmoEmbedder()
> tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
> vectors = elmo.embed_sentence(tokens)

> assert(len(vectors) == 3) # one for each layer in the ELMo output
> assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens

> import scipy
> vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])
> scipy.spatial.distance.cosine(vectors[2][3], vectors2[2][3]) # cosine distance between "apple" and "carrot" in the last layer
0.18020617961883545
```

## Using ELMo with existing `allennlp` models

In the simplest case, adding ELMo to an existing model is a simple
configuration change.  We provide a `TokenEmbedder` that accepts
character ids as input, runs the deep biLM and computes the ELMo representations
via a learned weighted combination.
Note that this simple case only includes one layer of ELMo representation
in the final model.
In some case (e.g. SQuAD and SNLI) we found that including multiple layers improved performance.  Multiple layers require code changes (see below).

We will use existing SRL model [configuration file](../../training_config/semantic_role_labeler.jsonnet) as an example to illustrate the changes.  Without ELMo, it uses 100 dimensional pre-trained GloVe vectors.

To add ELMo, there are three relevant changes.  First, modify the `text_field_embedder` section by adding an `elmo` section as follows:

```json
"text_field_embedder": {
  "tokens": {
    "type": "embedding",
    "embedding_dim": 100,
    "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
    "trainable": true
  },
  "elmo": {
    "type": "elmo_token_embedder",
    "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
    "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
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


# Reproducing the results from <i>Deep contextualized word representations</i>

This section provides details on reproducing the results in Table 1
of the [ELMo paper](https://www.aclweb.org/anthology/N18-1202).

For context, all of the experiments for the ELMo paper were done before AllenNLP existed, and almost all of the models in AllenNLP are re-implementations of things that were typically originally written in tensorflow code (the SRL model is the only exception).
In some cases, we haven't had the resources to tune the AllenNLP implementations to match the existing performance numbers yet; if you are able to do this for some of the models and submit back a tuned model, we (and many others) would greatly appreciate it.

For the tasks in Table 1, this table lists the corresponding AllenNLP config files in cases where we have a re-implementation, and notes about reproducing the results in cases where we do not.
The config files are in the [training_config/](../../training_config) folder.

|Task |    Configs |  Notes
|-----|------------|-------|
|SQuAD|   N/A | The SQuAD model is from [Clark and Gardner, 2018](https://www.semanticscholar.org/paper/Simple-and-Effective-Multi-Paragraph-Reading-Clark-Gardner/b95f7399861dd08d4f057bcbe2d6e21a9c543ddc). Tensorflow code to reproduce the results is [here](https://github.com/allenai/document-qa/tree/master/docqa/elmo).|
|SNLI| esim.jsonnet / esim_elmo.jsonnet  | This configuration is modified slightly from the one used in the ELMo paper, but performance is comparable. AllenNLP re-implementation has test accuracy 88.5% (original 88.7 +/- 0.17).  See the comment in esim_elmo.jsonnet for more details.|
|SRL | semantic_role_labeler.jsonnet /  semantic_role_labeler_elmo.jsonnet    | There's also a config that uses the ELMo trained on 5.5B tokens. Note: the SRL model is exceedingly slow to train. There is a faster version with a custom CUDA kernel available, but it is being depreciated and is incompatible with newer allennlp releases.  See [this discussion](https://github.com/allenai/allennlp/pull/1626#issuecomment-416697726) for details.  Also, the SRL metric implementation (`SpanF1Measure`) does not exactly track the output of the official PERL script (is typically 1-1.5 F1 below), and reported results used the official evaluation script.|
|Coref | coref.jsonnet  / NA | The allennlp re-implementation is missing some features of the original tensorflow version and performance is a few percent below the original result. See [Tensorflow code](https://github.com/kentonl/e2e-coref) for running the original experiments (baseline and with ELMo) and extentions reported in [Lee et al. 2018, "Higher-order Coreference Resolution with Coarse-to-fine Inference"](https://arxiv.org/abs/1804.05392).|
|NER | ner.jsonnet  / ner_elmo.jsonnet  | AllenNLP baseline has F1 of 89.91 +/- 0.35 (Keras original is 90.15). AllenNLP with ELMo single run F1 is 92.51 (original 92.22 +/- 0.10), see ner_elmo.jsonnnet for details.|
|SST-5 | biattentive_classification_network.jsonnet / biattentive_classification_network_elmo.jsonnet  | AllenNLP baseline single random seed test accuracy is 51.3 (original 51.4), with ELMo accuracy is 54.7 (original is 54.7 +/- 0.5).  See biattentive_classification_network_elmo.jsonnet for details.|

