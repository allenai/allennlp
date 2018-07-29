"""
The ``generate_subexperiments`` subcommand can be used to generate
configs of subexperiments from a base/root experiment. It requires a
root config file, sub experiment generator config file and directory
to store the generated subexperiment configs.

.. code-block:: bash

    $ allennlp generate-subexperiments --help
    usage: allennlp generate-subexperiments [-h]
                                            root_experiment_file generator_file
                                            subexperiments_dir

    Generate subexperiment configs form root experiment config.

    positional arguments:
      root_experiment_file  path to root experiment config file
      generator_file        path to sub-experiment generator config file
      subexperiments_dir    output directory to store generated config files

    optional arguments:
      -h, --help            show this help message and exit
"""
import itertools
from copy import deepcopy
import shutil
from typing import Union, Sequence, Iterable
from collections import OrderedDict
import argparse
import os
import json
import ast
import logging

import _jsonnet

from allennlp.commands.subcommand import Subcommand
from allennlp.common.params import Params, parse_overrides, with_fallback
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Change:
    """
    Encodes a single change from source config to updated config which doesn't need to be
    internally combined. Eg. change num_dim of embeddings to 300, which requires 'model.
    text_field_embedder.tokens.pretrained_file' set to '/pretrained_files/glove.6B.300d.
    txt.gz' and 'model.text_field_embedder.embedding_dim' set to 300. Change has tuple of
    keys and tuple of values of same size. Corresponding key will be set to corresponding
    value when this Change is executed. Key is string with dot (.) to represent nesting
    (as in jsonnet). Value are any valid jsonnet expression strings with access to extra
    variable ``current``, which refers to state of source config before change execution.

    Parameters
    ----------
    key_tuple : ``Sequence[str]``,
        tuple of jsonnet field references expressed in strings.
        eg ("trainer.num_epochs", "trainer.batch_size").
    value_tuple : ``Sequence[str]``,
        tuple of jsonnet expressions of expressed in strings of same size as `key_tuple`.
        Corresponding key will be set to corresponding value when this Change is executed.
        Value must be valid jsonnet expression with access to ``current`` variable refer
        ing to state of source config before change execution.
        Eg 1. `key_tuple`: ("50", "64") for `key_tuple`: ("trainer.num_epochs", "trainer.batch_size").
        Eg 2. `key_tuple`: ("model.tuple_feedforward.hidden_dims") and value tuple:
        ("[200 for _ in std.range(1, current.model.tuple_feedforward.num_layers)]")
    key_name : ``str``, optional (joined ``key_tuple``s with '-')
        This name will be used to refer to key in experiment id which has this change exectued.
        key can often be very large, key_name provides good short name that is easy to parse for
        humans. Eg. You can keep 'bidrectional' if you set 'model.seq2vec_encoder.bidirectional
        in this change.
    value_name : ``str``, optional (joined ``value_tuple``s with '-')
        This name will be used to refer to value in experiment id which has this change exectued.
        Eg. You can keep 'true' if you set 'model.seq2vec_encoder.bidirectional" to True in this
        change. This key,value change will be referred as ``key_name``=`value_name` in the
        experiment id.
    """
    def __init__(self,
                 key_tuple: Sequence[str],
                 value_tuple: Sequence[str],
                 key_name: Union[str, int, bool] = None,
                 value_name: Union[str, int, bool] = None) -> None:
        if not isinstance(key_tuple, (list, tuple)):
            raise ConfigurationError("key_tuple: '{}' must be tuple / list".format(key_tuple))
        if not isinstance(value_tuple, (list, tuple)):
            raise ConfigurationError("value_tuple: '{}' must be tuple / list".format(value_tuple))
        if len(key_tuple) != len(value_tuple):
            raise ConfigurationError("key_tuple: '{}' and value_tuple: '{}' "
                                     "both must be tuples of same size.".format(key_tuple, value_tuple))
        self.key_tuple = key_tuple
        self.value_tuple = value_tuple
        self.key_name = key_name or "_".join(key_tuple)
        self.value_name = value_name or "_".join(value_tuple)
        valid_key_name = any(isinstance(self.key_name, _type) for _type in [str, int, bool])
        valid_value_name = any(isinstance(self.value_name, _type) for _type in [str, int, bool])
        if not valid_key_name or not valid_value_name:
            raise ConfigurationError("Value name and Key name must be string / int / bool")

    def __str__(self) -> str:
        return "{}={}".format(self.key_name, self.value_name)

    def execute(self, source_config: Params) -> Params:
        """
        Takes in source_config and executes the change that this instance encodes.
        """
        define_current_var = 'local current = {} ;'.format(json.dumps(source_config.as_dict(quiet=True)))
        overrides_dict = {}
        for key, value in zip(self.key_tuple, self.value_tuple):
            try:
                value_evaluated = _jsonnet.evaluate_snippet("", define_current_var+value).strip()
            except:
                raise ConfigurationError("Following is not valid jsonnet expression:\n"
                                         "{}".format(define_current_var+value))
            # value_evaluated can be int / string / dict / bool. But it will be represented
            # in terms of json objects. Eg. if value_evaluated is True, it will be 'true'.
            # Since it may not be dict, json.loads does following is required.
            for json_str, py_str in [('null', 'None'), ('true', 'True'), ('false', 'False')]:
                value_evaluated = value_evaluated.replace(json_str, py_str)
            try:
                value_evaluated = ast.literal_eval(value_evaluated)
            except:
                raise ConfigurationError("{} could not be parsed as python expression.".format(value_evaluated))
            overrides_dict[key] = value_evaluated
        overrides_dict = parse_overrides(json.dumps(overrides_dict))
        changed_config = with_fallback(preferred=overrides_dict,
                                       fallback=source_config.as_dict(quiet=True))
        return Params(changed_config)

    @classmethod
    def from_params(cls, change_params: Params) -> 'Change':
        key_tuple = change_params.pop("key_tuple")
        value_tuple = change_params.pop("value_tuple")
        default_key_name = "_".join(key_tuple)
        default_value_name = "_".join(value_tuple)
        key_name = change_params.pop("key_name", default_key_name)
        value_name = change_params.pop("value_name", default_value_name)
        change_params.assert_empty('Change')
        return cls(key_tuple=key_tuple, value_tuple=value_tuple,
                   key_name=key_name, value_name=value_name)


class GroupOfChanges:
    """
    Encodes group of similar Changes to be used one at a time.
    Eg. group for trying embedding sizes from 100, 200 and 300.

    Parameters
    ----------
    changes : ``Sequence[Change]``,
        List/Tuple of changes with same ``key_tuple``
    """
    def __init__(self, changes: Sequence[Change] = ()) -> None:
        if len({tuple(change.key_tuple) for change in changes}) > 1:
            raise ConfigurationError("A GroupOfChanges must have single key_tuple.")
        self.changes = changes

    def __str__(self) -> str:
        return ','.join([str(change) for change in self.changes])

    @classmethod
    def from_params(cls, change_params: Params) -> 'GroupOfChanges':
        key_tuple = change_params.pop("key_tuple")
        value_tuples = change_params.pop("value_tuples")
        default_key_name = "_".join([str(key) for key in key_tuple])
        default_value_names = ["_".join([str(value) for value in value_tuple])
                               for value_tuple in value_tuples]
        key_name = change_params.pop("key_name", default_key_name)
        value_names = change_params.pop("value_names", default_value_names)
        change_params.assert_empty('GroupOfChanges')

        if len(value_tuples) != len(value_names):
            raise ConfigurationError("Number of value tuples '{}' and corresponding "
                                     "value names '{}' must be same.".format(value_tuples, value_names))
        changes = [Change(key_tuple=key_tuple, value_tuple=value_tuple,
                          key_name=key_name, value_name=value_name)
                   for value_tuple, value_name in zip(value_tuples, value_names)]
        return cls(changes)


class SequenceOfChanges:
    """
    Encodes sequence of Changes to be executed in that order.
    Eg. Set model.seq2vec_encoder.bidirectional set to True, Set embedding dim to 300
    and Use glove embeddings of 300D.

    Parameters
    ----------
    changes : ``Sequence[Change]``,
        List/Tuple of changes
    """

    def __init__(self, changes: Sequence[Change] = ()) -> None:
        if len({tuple(change.key_tuple) for change in changes}) < len(changes):
            logger.info("Some changes have same 'key_tuple'. On executing "
                        "this sequence, only later change will retain.")
        self.changes = changes

    def __str__(self) -> str:
        return ','.join([str(change) for change in self.changes])

    def execute(self, source_config: Params) -> Params:
        """
        Applies each change in this SequenceOfChanges sequentially.
        """
        updated_config = deepcopy(source_config)
        for change in self.changes:
            updated_config = change.execute(updated_config)
        return updated_config

    @classmethod
    def from_params_list(cls, changes_params: Sequence[Params] = ()) -> 'SequenceOfChanges':
        return cls([Change.from_params(change_params) for change_params in changes_params])


class SubExperimentsGenerator:
    """
    Takes root experiment config, sequence of pre-combine changes, sequence of
    post-combine changes, and a list of groups of changes to be made combinations from.
    Combinations are generated from list of groups taking one from each group at a
    time. For each resultant combination (sequence of changes), changes are executed
    in following order: pre-combine changes, combination sequence, post-combine changes.
    Resulting generated experiment configs can be stored in directory by calling save.

    Parameters
    ----------
    root_config : ``Params``
        root/base experiment config to generate sub-experiments from.
    pre_combine_changes : ``SequenceOfChanges``
        Sequence of changes to be applied before combination.
    to_be_combined_changes : ``Sequence[GroupOfChanges]``
        List/Tuple of groups of changes to be combined across. One change will be
        taken from each group for one experiment.
    post_combine_changes : ``SequenceOfChanges``
        Sequence of changes to be applied after applying a combination of change.
    """
    def __init__(self,
                 root_config: Params,
                 pre_combine_changes: SequenceOfChanges = SequenceOfChanges(()),
                 to_be_combined_changes: Sequence[GroupOfChanges] = (),
                 post_combine_changes: SequenceOfChanges = SequenceOfChanges(())) -> None:
        self._root_config = root_config
        self._pre_combine_changes = pre_combine_changes
        self._to_be_combined_changes = to_be_combined_changes
        self._post_combine_changes = post_combine_changes
        self._subexperiments = self._generate()

    @classmethod
    def from_params(cls,
                    root_config: Params,
                    generator_config: Params) -> 'SubExperimentsGenerator':

        pre_combine_configs = generator_config.pop("pre_combine_changes", ())
        if not isinstance(pre_combine_configs, (list, tuple)):
            raise ConfigurationError("key 'pre_combine_changes' must be a list/tuple")
        pre_combine_changes = SequenceOfChanges.from_params_list(pre_combine_configs)

        post_combine_configs = generator_config.pop("post_combine_changes", ())
        if not isinstance(post_combine_configs, (list, tuple)):
            raise ConfigurationError("key 'post_combine_changes' must be a list/tuple")
        post_combine_changes = SequenceOfChanges.from_params_list(post_combine_configs)

        group_change_configs = generator_config.pop("combine_changes")
        to_be_combined_changes = [GroupOfChanges.from_params(group_change_config)
                                  for group_change_config in group_change_configs]

        generator_config.assert_empty('SubExperimentsGenerator generator_config')

        return cls(root_config=root_config,
                   pre_combine_changes=pre_combine_changes,
                   to_be_combined_changes=to_be_combined_changes,
                   post_combine_changes=post_combine_changes)

    def _generate_combinations_of_changes(self) -> Iterable[SequenceOfChanges]:
        """
        Makes combinations from groups of changes,by taking one change from each group
        for each combination. Encodes each combinations as SequenceOfChanges.
        """
        list_of_group_of_changes = [group_of_changes.changes
                                    for group_of_changes in self._to_be_combined_changes]
        for combination_changes in itertools.product(*list_of_group_of_changes):
            yield SequenceOfChanges(combination_changes)

    def _generate(self) -> OrderedDict:
        """
        Generates sub-experiments and returns ordered dict keyed by experiment name
        and valued by final experiment config params.
        """
        subexperiments: OrderedDict = OrderedDict({})
        combinations_of_changes = self._generate_combinations_of_changes()
        for combination in combinations_of_changes:
            experiment_config = self._pre_combine_changes.execute(self._root_config)
            experiment_config = combination.execute(experiment_config)
            experiment_config = self._post_combine_changes.execute(experiment_config)
            experiment_name = str(combination)
            subexperiments[experiment_name] = experiment_config
            logger.info("Generated: %s", experiment_name)
        logger.info("%s subexperiments generated", len(subexperiments)) # temporary
        return subexperiments

    def save(self,
             subexperiments_dir: str,
             root_name: str = "",
             order: bool = True) -> None:
        """ Stores the generated experiment configs in ``subexperiments_dir``."""
        if os.path.isdir(subexperiments_dir):
            force = input("The subexperiments_dir: '{}' exists. Delete? Y/N".format(subexperiments_dir))
            if 'n' in force.lower():
                exit("Terminated by user.")
            shutil.rmtree(subexperiments_dir)
        os.makedirs(subexperiments_dir)
        for index, subexperiment in enumerate(self._subexperiments.items()):
            subexperiment_name, subexperiment_config = subexperiment
            order_name = str(index+1) if order else ""
            file_name = ".".join([order_name, root_name, subexperiment_name, "json"])
            file_path = os.path.join(subexperiments_dir, file_name)
            subexperiment_config.to_file(file_path)


class GenerateSubexperiments(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Generate subexperiment configs form root experiment config.'''
        subparser = parser.add_parser(name, description=description, help='Train a model')
        subparser.add_argument('root_experiment_file', type=str,
                               help='path to root experiment config file')
        subparser.add_argument('generator_file', type=str,
                               help='path to sub-experiment generator config file')
        subparser.add_argument('subexperiments_dir', type=str,
                               help='output directory to store generated config files')
        subparser.set_defaults(func=generate_subexperiments_from_args)
        return subparser


def generate_subexperiments_from_args(args: argparse.Namespace):

    root_config = Params.from_file(args.root_experiment_file)
    generator_config = Params.from_file(args.generator_file)
    root_experiment_name = os.path.splitext(os.path.basename(args.root_experiment_file))[0]
    generator = SubExperimentsGenerator.from_params(root_config, generator_config)
    generator.save(args.subexperiments_dir, root_name=root_experiment_name)
