# allennlp.commands.test_install

The ``test-install`` subcommand verifies
an installation by running the unit tests.

.. code-block:: bash

    $ allennlp test-install --help
    usage: allennlp test-install [-h] [--run-all] [-k K]
                                 [--include-package INCLUDE_PACKAGE]

    Test that installation works by running the unit tests.

    optional arguments:
      -h, --help            show this help message and exit
      --run-all             By default, we skip tests that are slow or download
                            large files. This flag will run all tests.
      -k K                  Limit tests by setting pytest -k argument
      --include-package INCLUDE_PACKAGE
                            additional packages to include

