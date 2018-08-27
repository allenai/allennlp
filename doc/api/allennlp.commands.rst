allennlp.commands
=========================

These submodules contain the command line tools for things like
training and evaluating models. You probably don't want to call
most of them directly. Instead, just create a script that calls
``allennlp.commands.main()`` and it will automatically inherit
all of the subcommands in this module.

The included module ``allennlp.run`` is such a script:

.. code-block:: bash

    Run AllenNLP

    optional arguments:
      -h, --help  show this help message and exit

    Commands:
        train       Train a model
        configure   Generate a stub configuration
        evaluate    Evaluate the specified model + dataset
        predict     Use a trained model to make predictions.
        make-vocab  Create a vocabulary
        elmo        Use a trained model to make predictions.
        fine-tune   Continue training a model on a new dataset
        dry-run     Create a vocabulary, compute dataset statistics and other
                    training utilities.
        test-install
                    Run the unit tests.

However, it only knows about the models and classes that are
included with AllenNLP. Once you start creating custom models,
you'll need to make your own script which imports them and then
calls ``main()``.

.. automodule:: allennlp.commands
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: allennlp.commands.subcommand

.. automodule:: allennlp.commands.configure

.. automodule:: allennlp.commands.evaluate

.. automodule:: allennlp.commands.make_vocab

.. automodule:: allennlp.commands.predict

.. automodule:: allennlp.commands.train

.. automodule:: allennlp.commands.fine_tune

.. automodule:: allennlp.commands.elmo

.. automodule:: allennlp.commands.dry_run

.. automodule:: allennlp.commands.test_install
