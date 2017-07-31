# pylint: disable=no-self-use,invalid-name
import argparse
from unittest import TestCase

from allennlp.__main__ import main
from allennlp.commands.serve import add_serve_subparser, serve


class TestServe(TestCase):

    def test_add_serve(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_serve_subparser(subparsers)

        raw_args = ["serve",
                    "--backend",
                    "flask",
                    "--port",
                    "--8000"]

        args = parser.parse_args(raw_args)

        assert args.func == serve
        assert args.backend == "flask"
        assert args.port == "8000"

