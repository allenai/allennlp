import argparse
import json

from allennlp.service import server_sanic

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
    config = server_sanic.DEFAULT_CONFIG
    if args.config_file is not None:
        with args.config_file as fopen:
            config = json.loads(fopen.read())

    server_sanic.run(args.port, args.workers, config)
