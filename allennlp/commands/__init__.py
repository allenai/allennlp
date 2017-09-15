from typing import Dict
import argparse

from allennlp.commands.serve import add_subparser as add_serve_subparser
from allennlp.commands.predict import add_subparser as add_predict_subparser
from allennlp.commands.train import add_subparser as add_train_subparser
from allennlp.commands.evaluate import add_subparser as add_evaluate_subparser
from allennlp.common.checks import ensure_pythonhashseed_set

# a mapping from model `type` to the location of the trained model of that type
DEFAULT_MODELS = {
        'machine-comprehension': 'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.08.31.tar.gz', # pylint: disable=line-too-long
        'semantic-role-labeling': 'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2017.09.05.tar.gz', # pylint: disable=line-too-long
        'textual-entailment': 'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-2017.09.04.tar.gz' # pylint: disable=line-too-long
}

# a mapping from model `type` to the default Predictor for that type
DEFAULT_PREDICTORS = {
        'srl': 'semantic-role-labeling',
        'decomposable_attention': 'textual-entailment',
        'bidaf': 'machine-comprehension',
        'simple_tagger': 'simple-tagger'
}

def main(prog: str = None,
         model_overrides: Dict[str, str] = {},
         predictor_overrides: Dict[str, str] = {}) -> None:
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
    add_serve_subparser(subparsers, trained_models=trained_models, predictors=predictors)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
