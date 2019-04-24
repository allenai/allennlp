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
        elmo        Create word vectors using a pretrained ELMo model.
        fine-tune   Continue training a model on a new dataset
        dry-run     Create a vocabulary, compute dataset statistics and other
                    training utilities.
        find-lr     Find a learning rate range where loss decreases quickly
                    for the specified model and dataset.
        test-install
                    Run the unit tests.
        print-results
                    Print results from allennlp serialization directories to the
                    console.

However, it only knows about the models and classes that are
included with AllenNLP. Once you start creating custom models,
you'll need to make your own script which imports them and then
calls ``main()``.

.. toctree::
    allennlp.commands.subcommand
    allennlp.commands.configure
    allennlp.commands.evaluate
    allennlp.commands.make_vocab
    allennlp.commands.predict
    allennlp.commands.train
    allennlp.commands.fine_tune
    allennlp.commands.elmo
    allennlp.commands.dry_run
    allennlp.commands.find_learning_rate
    allennlp.commands.test_install
    allennlp.commands.print_results

.. automodule:: allennlp.commands
   :members:
   :undoc-members:
   :show-inheritance:
