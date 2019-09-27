"""
The ``configure`` subcommand launches a webapp that helps you
generate an AllenNLP configuration file.

.. code-block:: bash

    $ allennlp configure --help
    usage: allennlp configure [-h] [--port PORT]
                              [--include-package INCLUDE_PACKAGE]

    Run the configuration wizard

    optional arguments:
      -h, --help            show this help message and exit
      --port PORT           port to serve the wizard on (default = 8123)
      --include-package INCLUDE_PACKAGE
                            additional packages to include
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
    print(f"serving Config Explorer at http://localhost:{args.port}")
    http_server.serve_forever()
