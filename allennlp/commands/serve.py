import argparse

from allennlp.service import server_sanic

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
    subparser = parser.add_parser(
            'serve', description=description, help='Run the web service and demo.')

    subparser.add_argument('--port', type=int, default=8000)
    subparser.add_argument('--workers', type=int, default=1)

    subparser.set_defaults(func=serve)

    return subparser

def serve(args: argparse.Namespace) -> None:
    server_sanic.run(args.port, args.workers)
