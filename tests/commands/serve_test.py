# pylint: disable=no-self-use,invalid-name
import argparse
from unittest import TestCase

from allennlp.commands import DEFAULT_MODELS
from allennlp.commands.serve import add_subparser


class TestServe(TestCase):

    def test_add_serve(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        add_subparser(subparsers, trained_models=DEFAULT_MODELS)

        raw_args = ["serve",
                    "--port", "8000"]

        args = parser.parse_args(raw_args)

        assert args.func.__name__ == 'serve_inner'
        assert args.port == 8000
