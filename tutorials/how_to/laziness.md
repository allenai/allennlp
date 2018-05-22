# How to Train a Model with Lazy Data

(Here, by "lazy data", I mean a dataset that's generated
 instance-by-instance each time you iterate over it,
 possibly because it's too big to fit in memory,
 possibly because you want to sample your data each iteration,
 possibly because you want to start training immediately,
 possibly for some other reason I haven't thought of.)

There are basically four things you should do. This how-to will tell you
what they are. If you want to know the details, you should check out the
[detailed tutorial about laziness in AllenNLP](../getting_started/laziness.md).

## In the `YourDatasetReader` constructor

`YourDatasetReader` subclass should have a `lazy` parameter in its constructor
and `from_params` method, and pass that value to its superclass constructor.

([Detailed explanation](../getting_started/laziness.md#you-specify-laziness-in-the-datasetreader-constructor))

## In `YourDatasetReader._read()`

`YourDatasetReader._read()` should return a generator, not a list.

([Detailed explanation](../getting_started/laziness.md#laziness-in-yourdatasetreader_read))

## In the `dataset_reader` section of your experiment configuration

You should specify `'lazy': true` in the `dataset_reader` section of your experiment config.

([Detailed explanation](../getting_started/laziness.md#laziness-in-experimentjson))

## In the `iterator` section of your experiment configuration

You should specify `'max_instances_in_memory'` as large as you can
in the `iterator` section of your experiment config.
(You don't _have_ to do this, but it's a good idea,
 especially if you want to use a `BucketIterator`.)

([Detailed explanation](../getting_started/laziness.md#laziness-in-dataiterator))
