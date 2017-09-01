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
import json

from allennlp.service import server_sanic

DEFAULT_CONFIG = {
        'machine-comprehension': 'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.08.31.tar.gz', # pylint: disable=line-too-long
        'semantic-role-labeling': 'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2017.08.28.tar.gz', # pylint: disable=line-too-long
        'textual-entailment': 'tests/fixtures/decomposable_attention/serialization/model.tar.gz' # pylint: disable=line-too-long
}

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
    subparser = parser.add_parser(
            'serve', description=description, help='Run the web service and demo.')

    subparser.add_argument('--port', type=int, default=8000)
    subparser.add_argument('--workers', type=int, default=1)
    subparser.add_argument('--config-file', type=argparse.FileType('r'), default=None,
                           help="path to a JSON file specifying the configuration for the models")

    subparser.set_defaults(func=serve)

    return subparser

def serve(args: argparse.Namespace) -> None:
    # Read a JSON configuration file, if specified
    config = DEFAULT_CONFIG
    if args.config_file is not None:
        with args.config_file as fopen:
            config = json.loads(fopen.read())

    server_sanic.run(args.port, args.workers, config)
