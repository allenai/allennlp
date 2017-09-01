allennlp.commands
=========================

These submodules contain the command line tools for things like
training and evaluating models. You probably don't want to call
most of them directly. Instead, just create a script that calls
``allennlp.commands.main()`` and it will automatically inherit
all of the subcommands in this module.

The included module ``allennlp.run`` is such a script:

.. code-block:: bash

    $ python -m allennlp.run --help
    usage: run [command]

    Run AllenNLP

    optional arguments:
    -h, --help  show this help message and exit

    Commands:

        predict   Use a trained model to make predictions.
        train     Train a model
        serve     Run the web service and demo.
        evaluate  Evaluate the specified model + dataset

However, it only knows about the models and classes that are
included with AllenNLP. Once you start creating custom models,
you'll need to make your own script which imports them and then
calls ``main()``.

.. toctree::
    allennlp.commands.evaluate
    allennlp.commands.predict
    allennlp.commands.serve
    allennlp.commands.train

.. automodule:: allennlp.commands
   :members:
   :undoc-members:
   :show-inheritance:
