"""
The `checklist` subcommand allows you to conduct behavioural
testing for your model's predictions using a trained model and its
[`Predictor`](../predictors/predictor.md#predictor) wrapper.

It is based on the optional checklist package; if this is not
available, the command will be replaced by a dummy.
"""

import argparse
import logging

from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)

try:
    from allennlp.commands._checklist_internal import CheckList
except ImportError:
    # create dummy command that tells users how to
    # install the necessary dependency

    def _dummy_output(args: argparse.Namespace):
        logger.info(
            "The checklist integration of allennlp is optional; if you're using conda, "
            "it can be installed with `conda install allennlp-checklist`, "
            "otherwise use `pip install allennlp[checklist]`."
        )

    # need to work around https://github.com/python/mypy/issues/1153
    @Subcommand.register("checklist")
    class CheckList(Subcommand):  # type: ignore
        def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
            description = """Dummy command because checklist is not installed."""
            subparser = parser.add_parser(
                self.name,
                description=description,
                help="Run a trained model through a checklist suite.",
            )
            subparser.set_defaults(func=_dummy_output)
            return subparser
