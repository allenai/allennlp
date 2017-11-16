"""
The ``serve`` subcommand launches a server
that exposes trained models via a REST API,
and that includes a web interface for exploring
their predictions.

.. code-block:: bash

    $ python -m allennlp.run serve --help
    usage: run [command] serve [-h] [--port PORT] [--workers WORKERS]
                            [--config-file CONFIG_FILE]

    Run the web service, which provides an HTTP API as well as a web demo.

    optional arguments:
    -h, --help            show this help message and exit
    --port PORT
    --workers WORKERS
    --config-file CONFIG_FILE
                            path to a JSON file specifying the configuration for
                            the models
"""

import argparse
from typing import Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.service import server_sanic
from allennlp.service.predictors import DemoModel

class Serve(Subcommand):
    def __init__(self, trained_models: Dict[str, DemoModel]) -> None:
        self.trained_models = trained_models

    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
        subparser = parser.add_parser(
                name, description=description, help='Run the web service and demo.')

        subparser.add_argument('--port', type=int, default=8000)
        subparser.add_argument('--workers', type=int, default=1)

        subparser.set_defaults(func=_serve(self.trained_models))

        return subparser

def _serve(trained_models: Dict[str, DemoModel]):
    def serve_inner(args: argparse.Namespace) -> None:
        server_sanic.run(args.port, args.workers, trained_models)

    return serve_inner
