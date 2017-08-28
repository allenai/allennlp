import argparse
import json

import logging
logger = logging.getLogger(__name__)

from allennlp.service import server_sanic

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
    subparser = parser.add_parser(
            'serve', description=description, help='Run the web service and demo.')

    subparser.add_argument('--port', type=int, default=8000)
    subparser.add_argument('--workers', type=int, default=1)
    subparser.add_argument('--configfile', type=argparse.FileType('r'), default=None, help="path to a JSON file specifying the configuration for the models")

    subparser.set_defaults(func=serve)

    return subparser

def serve(args: argparse.Namespace) -> None:
    # Read a JSON configuration file, if specified
    config = server_sanic.default_config
    if args.configfile is not None:
        with args.configfile as f:
            config = json.loads(f.read())

    server_sanic.run(args.port, args.workers, config)
