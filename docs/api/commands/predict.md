# allennlp.commands.predict

The ``predict`` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
:class:`~allennlp.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict --help
    usage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE] [--use-dataset-reader]
                            [--dataset-reader-choice {train,validation}]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
      archive_file          the archived model to make predictions with
      input_file            path to or url of the input file

    optional arguments:
      -h, --help            show this help message and exit
      --output-file OUTPUT_FILE
                            path to output file
      --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
      --batch-size BATCH_SIZE
                            The batch size to use for processing
      --silent              do not print output to stdout
      --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
      --use-dataset-reader  Whether to use the dataset reader of the original
                            model to load Instances. The validation dataset reader
                            will be used if it exists, otherwise it will fall back
                            to the train dataset reader. This behavior can be
                            overridden with the --dataset-reader-choice flag.
      --dataset-reader-choice {train,validation}
                            Indicates which model dataset reader to use if the
                            --use-dataset-reader flag is set. (default =
                            validation)
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --predictor PREDICTOR
                            optionally specify a specific predictor to use
      --include-package INCLUDE_PACKAGE
                            additional packages to include

