# allennlp.commands.evaluate

The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--batch-size BATCH_SIZE]
                             [--batch-weight-key BATCH_WEIGHT_KEY]
                             [--extend-vocab]
                             [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
      archive_file          path to an archived trained model
      input_file            path to the file containing the evaluation data

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --batch-size BATCH_SIZE
                            If non-empty, the batch size to use during evaluation.
      --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
      --extend-vocab        if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
      --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
      --include-package INCLUDE_PACKAGE
                            additional packages to include

