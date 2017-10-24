from typing import Dict
import argparse

from allennlp.commands.serve import add_subparser as add_serve_subparser
from allennlp.commands.predict import add_subparser as add_predict_subparser
from allennlp.commands.train import add_subparser as add_train_subparser
from allennlp.commands.evaluate import add_subparser as add_evaluate_subparser
from allennlp.common.checks import ensure_pythonhashseed_set

# a mapping from predictor `type` to the location of the trained model of that type
DEFAULT_MODELS = {
        'machine-comprehension': 'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz', # pylint: disable=line-too-long
        'semantic-role-labeling': 'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2017.09.05.tar.gz', # pylint: disable=line-too-long
        'textual-entailment': 'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz' # pylint: disable=line-too-long
}

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'simple_tagger': 'simple-tagger',
        'crf_tagger': 'crf-tagger',
        'ontoemma': 'ontoemma'
}

def main(prog: str = None,
         model_overrides: Dict[str, str] = {},
         predictor_overrides: Dict[str, str] = {}) -> None:
    """
    The :mod:``allennlp.run`` command only knows about the registered classes
    in the ``allennlp`` codebase. In particular, once you start creating your own
    ``Model``s and so forth, it won't work for them. However, ``allennlp.run`` is
    simply a wrapper around this function. To use the command line interface with your
    own custom classes, just create your own script that imports all of the classes you want
    and then calls ``main()``.

    The default models for ``serve`` and the default predictors for ``predict`` are
    defined above. If you'd like to add more or use different ones, the
    ``model_overrides`` and ``predictor_overrides`` arguments will take precedence over the defaults.
    """
    # pylint: disable=dangerous-default-value
    ensure_pythonhashseed_set()

    parser = argparse.ArgumentParser(description="Run AllenNLP", usage='%(prog)s [command]', prog=prog)
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    trained_models = {**DEFAULT_MODELS, **model_overrides}
    predictors = {**DEFAULT_PREDICTORS, **predictor_overrides}

    # Add sub-commands
    add_train_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    add_predict_subparser(subparsers, predictors=predictors)
    add_serve_subparser(subparsers, trained_models=trained_models)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
