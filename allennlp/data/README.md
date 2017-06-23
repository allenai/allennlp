# Data

This module contains code for processing data.  There's a `DataIndexer`, whose job it is to convert
from strings to word (or character) indices suitable for use with an embedding matrix.  There's
code to load pre-trained embeddings from a file, to tokenize sentences, and, most importantly, to
convert training and testing examples into numpy arrays that can be used with Keras.

The most important thing to understand about the data processing code is the `Dataset` object.  A
`Dataset` is a collection of `Instances`, which are the individual examples used for training and
testing.  `Dataset` has two subclasses: `TextDataset`, which contains `Instances` with raw strings
and can be read directly from a file, and `IndexedDataset`, which contains `Instances` whose raw
strings have been converted to word (or character) indices.  The `IndexedDataset` has methods for
padding sequences to a consistent length, so that models can be compiled, and for converting the
`Instances` to numpy arrays.  The file formats read by `TextDataset`, and the format of the numpy
arrays produced by `IndexedDataset`, are determined by the underlying `Instance` type used by the
`Dataset`.  See the `instances` module for more detail on this.
