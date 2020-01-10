# allennlp.commands.print_results

The ``print-results`` subcommand allows you to print results from multiple
allennlp serialization directories to the console in a helpful csv format.

.. code-block:: bash

   $ allennlp print-results --help
    usage: allennlp print-results [-h] [-k KEYS [KEYS ...]] [-m METRICS_FILENAME]
                                  [--include-package INCLUDE_PACKAGE]
                                  path

    Print results from allennlp training runs in a helpful CSV format.

    positional arguments:
      path                  Path to recursively search for allennlp serialization
                            directories.

    optional arguments:
      -h, --help            show this help message and exit
      -k KEYS [KEYS ...], --keys KEYS [KEYS ...]
                            Keys to print from metrics.json.Keys not present in
                            all metrics.json will result in "N/A"
      -m METRICS_FILENAME, --metrics-filename METRICS_FILENAME
                            Name of the metrics file to inspect. (default =
                            metrics.json)
      --include-package INCLUDE_PACKAGE
                            additional packages to include

## print_results_from_args
```python
print_results_from_args(args:argparse.Namespace)
```

Prints results from an ``argparse.Namespace`` object.

