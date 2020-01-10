# allennlp.commands.elmo

The ``elmo`` subcommand allows you to make bulk ELMo predictions.

Given a pre-processed input text file, this command outputs the internal
layers used to compute ELMo representations to a single (potentially large) file.

The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (<https://h5py.readthedocs.io/en/latest/>) where, with the --all flag, each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.

For information, see "Deep contextualized word representations", Peters et al 2018.
https://arxiv.org/abs/1802.05365

.. code-block:: console

   $ allennlp elmo --help
    usage: allennlp elmo [-h] (--all | --top | --average)
                         [--vocab-path VOCAB_PATH] [--options-file OPTIONS_FILE]
                         [--weight-file WEIGHT_FILE] [--batch-size BATCH_SIZE]
                         [--file-friendly-logging] [--cuda-device CUDA_DEVICE]
                         [--forget-sentences] [--use-sentence-keys]
                         [--include-package INCLUDE_PACKAGE]
                         input_file output_file

    Create word vectors using ELMo.

    positional arguments:
      input_file            The path to the input file.
      output_file           The path to the output file.

    optional arguments:
      -h, --help            show this help message and exit
      --all                 Output all three ELMo vectors.
      --top                 Output the top ELMo vector.
      --average             Output the average of the ELMo vectors.
      --vocab-path VOCAB_PATH
                            A path to a vocabulary file to generate.
      --options-file OPTIONS_FILE
                            The path to the ELMo options file. (default = https://
                            allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048c
                            nn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options
                            .json)
      --weight-file WEIGHT_FILE
                            The path to the ELMo weight file. (default = https://a
                            llennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cn
                            n_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.
                            hdf5)
      --batch-size BATCH_SIZE
                            The batch size to use. (default = 64)
      --file-friendly-logging
                            outputs tqdm status on separate lines and slows tqdm
                            refresh rate.
      --cuda-device CUDA_DEVICE
                            The cuda_device to run on. (default = -1)
      --forget-sentences    If this flag is specified, and --use-sentence-keys is
                            not, remove the string serialized JSON dictionary that
                            associates sentences with their line number (its HDF5
                            key) that is normally placed in the
                            "sentence_to_index" HDF5 key.
      --use-sentence-keys   Normally a sentence's line number is used as the HDF5
                            key for its embedding. If this flag is specified, the
                            sentence itself will be used as the key.
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## Elmo
```python
Elmo(self, /, *args, **kwargs)
```

Note that ELMo maintains an internal state dependent on previous batches.
As a result, ELMo will return differing results if the same sentence is
passed to the same ``Elmo`` instance multiple times.

See https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md for more details.

## ElmoEmbedder
```python
ElmoEmbedder(self, options_file:str='https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', weight_file:str='https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', cuda_device:int=-1) -> None
```

### batch_to_embeddings
```python
ElmoEmbedder.batch_to_embeddings(self, batch:List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]
```

Parameters
----------
batch : ``List[List[str]]``, required
    A list of tokenized sentences.

Returns
-------
    A tuple of tensors, the first representing activations (batch_size, 3, num_timesteps, 1024) and
the second a mask (batch_size, num_timesteps).

### embed_sentence
```python
ElmoEmbedder.embed_sentence(self, sentence:List[str]) -> numpy.ndarray
```

Computes the ELMo embeddings for a single tokenized sentence.

Please note that ELMo has internal state and will give different results for the same input.
See the comment under the class definition.

Parameters
----------
sentence : ``List[str]``, required
    A tokenized sentence.

Returns
-------
A tensor containing the ELMo vectors.

### embed_batch
```python
ElmoEmbedder.embed_batch(self, batch:List[List[str]]) -> List[numpy.ndarray]
```

Computes the ELMo embeddings for a batch of tokenized sentences.

Please note that ELMo has internal state and will give different results for the same input.
See the comment under the class definition.

Parameters
----------
batch : ``List[List[str]]``, required
    A list of tokenized sentences.

Returns
-------
    A list of tensors, each representing the ELMo vectors for the input sentence at the same index.

### embed_sentences
```python
ElmoEmbedder.embed_sentences(self, sentences:Iterable[List[str]], batch_size:int=64) -> Iterable[numpy.ndarray]
```

Computes the ELMo embeddings for a iterable of sentences.

Please note that ELMo has internal state and will give different results for the same input.
See the comment under the class definition.

Parameters
----------
sentences : ``Iterable[List[str]]``, required
    An iterable of tokenized sentences.
batch_size : ``int``, required
    The number of sentences ELMo should process at once.

Returns
-------
    A list of tensors, each representing the ELMo vectors for the input sentence at the same index.

### embed_file
```python
ElmoEmbedder.embed_file(self, input_file:IO, output_file_path:str, output_format:str='all', batch_size:int=64, forget_sentences:bool=False, use_sentence_keys:bool=False) -> None
```

Computes ELMo embeddings from an input_file where each line contains a sentence tokenized by whitespace.
The ELMo embeddings are written out in HDF5 format, where each sentence embedding
is saved in a dataset with the line number in the original file as the key.

Parameters
----------
input_file : ``IO``, required
    A file with one tokenized sentence per line.
output_file_path : ``str``, required
    A path to the output hdf5 file.
output_format : ``str``, optional, (default = "all")
    The embeddings to output.  Must be one of "all", "top", or "average".
batch_size : ``int``, optional, (default = 64)
    The number of sentences to process in ELMo at one time.
forget_sentences : ``bool``, optional, (default = False).
    If use_sentence_keys is False, whether or not to include a string
    serialized JSON dictionary that associates sentences with their
    line number (its HDF5 key). The mapping is placed in the
    "sentence_to_index" HDF5 key. This is useful if
    you want to use the embeddings without keeping the original file
    of sentences around.
use_sentence_keys : ``bool``, optional, (default = False).
    Whether or not to use full sentences as keys. By default,
    the line numbers of the input file are used as ids, which is more robust.

