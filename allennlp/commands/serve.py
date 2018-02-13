"""
The ``serve`` subcommand launches a server
that exposes trained models via a REST API,
and that includes a web interface for exploring
their predictions.

.. code-block:: bash

    $ python -m allennlp.run serve --help
    usage: python -m allennlp.run serve [-h] [--port PORT]

    Run the web service, which provides an HTTP API as well as a web demo.

    optional arguments:
    -h, --help   show this help message and exit
    --port PORT
"""

import argparse
from typing import Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.service import server_flask as server
from allennlp.service.predictors import DemoModel

# This maps from the name of the task
# to the ``DemoModel`` indicating the location of the trained model
# and the type of the ``Predictor``.  This is necessary, as you might
# have multiple models (for example, a NER tagger and a POS tagger)
# that have the same ``Predictor`` wrapper. The corresponding model
# will be served at the `/predict/<name-of-task>` API endpoint.
DEFAULT_MODELS = {
        'machine-comprehension': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz',  # pylint: disable=line-too-long
                'machine-comprehension'
        ),
        'semantic-role-labeling': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2017.09.05.tar.gz', # pylint: disable=line-too-long
                'semantic-role-labeling'
        ),
        'textual-entailment': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz',  # pylint: disable=line-too-long
                'textual-entailment'
        ),
        'coreference-resolution': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',  # pylint: disable=line-too-long
                'coreference-resolution'
        ),
        'named-entity-recognition': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2017.11.15.tar.gz',  # pylint: disable=line-too-long
                'sentence-tagger'
        )
}


class Serve(Subcommand):
    def __init__(self, model_overrides: Dict[str, DemoModel] = {}) -> None:
        # pylint: disable=dangerous-default-value
        self.trained_models = {**DEFAULT_MODELS, **model_overrides}

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
        subparser = parser.add_parser(
                name, description=description, help='Run the web service and demo.')

        subparser.add_argument('--port', type=int, default=8000)

        subparser.set_defaults(func=_serve(self.trained_models))

        return subparser

def _serve(trained_models: Dict[str, DemoModel]):
    def serve_inner(args: argparse.Namespace) -> None:
        server.run(args.port, trained_models)

    return serve_inner
