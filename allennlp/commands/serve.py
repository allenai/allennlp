import argparse

def add_subparser(parser: argparse._SubParsersAction) -> argparse.ArgumentParser:  # pylint: disable=protected-access
    description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
    subparser = parser.add_parser(
            'serve', description=description, help='Run the web service and demo.')

    subparser.add_argument('--backend', metavar='backend', type=str, choices=['flask', 'sanic'], default='flask',
                           help='The backend for the web service.')
    subparser.add_argument('--port', type=int, default=8000)

    subparser.set_defaults(func=serve)

    return subparser

def serve(args: argparse.Namespace) -> None:
    if args.backend == 'flask':
        from allennlp.service import server_flask
        server_flask.run(args.port)
    elif args.backend == 'sanic':
        from allennlp.service import server_sanic
        server_sanic.run(args.port)
    else:
        raise Exception("Unsupported backend: " + args.backend)
