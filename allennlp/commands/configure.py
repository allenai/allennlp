"""
The ``configure`` subcommand generates a stub configuration for
the specified class (or for the top level configuration if no class specified).

.. code-block:: bash

    $ allennlp configure --help
    usage: allennlp configure [-h] [class]

    Generate a configuration stub for a specific class (or for config as a whole if [class] is omitted).

    positional arguments:
    class

    optional arguments:
    -h, --help            show this help message and exit
"""

import argparse

from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.commands.subcommand import Subcommand
from allennlp.service.config_explorer import make_app


class Configure(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the configuration wizard'''
        subparser = parser.add_parser(
                name, description=description, help='Run the configuration wizard.')

        subparser.add_argument('--port', type=int, default=8123, help='port to serve the wizard on')
        subparser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')
        subparser.set_defaults(func=_run_wizard)

        return subparser

def _run_wizard(args: argparse.Namespace) -> None:
    app = make_app(args.include_package)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"serving Config Explorer on port {args.port}")
    http_server.serve_forever()
