"""
The ``serve`` subcommand launches a server
that exposes trained models via a REST API,
and that includes a web interface for exploring
their predictions.

.. code-block:: bash

    $ allennlp serve --help
    usage: allennlp serve [-h] [--port PORT]

    Run the web service, which provides an HTTP API as well as a web demo.

    optional arguments:
    -h, --help   show this help message and exit
    --port PORT
"""

import argparse

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
                'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.02.27.tar.gz', # pylint: disable=line-too-long
                'semantic-role-labeling'
        ),
        'textual-entailment': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz',  # pylint: disable=line-too-long
                'textual-entailment'
        ),
        'coreference-resolution': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',  # pylint: disable=line-too-long
                'coreference-resolution'
        ),
        'named-entity-recognition': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz',  # pylint: disable=line-too-long
                'sentence-tagger'
        ),
        'constituency-parsing': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',  # pylint: disable=line-too-long
                'constituency-parser'
        )
}


class Serve(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
        subparser = parser.add_parser(
                name, description=description, help='Run the web service and demo.')

        subparser.add_argument('--port', type=int, default=8000)

        subparser.set_defaults(func=_serve)

        return subparser

def _serve(args: argparse.Namespace):
    server.run(args.port, DEFAULT_MODELS)
