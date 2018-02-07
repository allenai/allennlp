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
        train     Train a model
        evaluate  Evaluate the specified model + dataset
        predict   Use a trained model to make predictions.
        serve     Run the web service and demo.
        make-vocab
                  Create a vocabulary
        elmo      Use a trained model to make predictions.

However, it only knows about the models and classes that are
included with AllenNLP. Once you start creating custom models,
you'll need to make your own script which imports them and then
calls ``main()``.

.. toctree::
    allennlp.commands.subcommand
    allennlp.commands.evaluate
    allennlp.commands.make_vocab
    allennlp.commands.predict
    allennlp.commands.serve
    allennlp.commands.train
    allennlp.commands.elmo

.. automodule:: allennlp.commands
   :members:
   :undoc-members:
   :show-inheritance:
