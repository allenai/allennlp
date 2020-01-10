# allennlp.commands.dry_run

The ``dry-run`` command creates a vocabulary, informs you of
dataset statistics and other training utilities without actually training
a model.

.. code-block:: bash

    $ allennlp dry-run --help
    usage: allennlp dry-run [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                            [--include-package INCLUDE_PACKAGE]
                            param_path

    Create a vocabulary, compute dataset statistics and other training utilities.

    positional arguments:
      param_path            path to parameter file describing the model and its
                            inputs

    optional arguments:
      -h, --help            show this help message and exit
      -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the output of the dry run.
      -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## dry_run_from_args
```python
dry_run_from_args(args:argparse.Namespace)
```

Just converts from an ``argparse.Namespace`` object to params.

