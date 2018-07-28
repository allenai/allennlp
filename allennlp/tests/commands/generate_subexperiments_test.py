# pylint: disable=no-self-use,protected-access
import os
from copy import deepcopy
import argparse

import pytest

from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.commands.generate_subexperiments import generate_subexperiments_from_args, \
    Change, GroupOfChanges, SequenceOfChanges, SubExperimentsGenerator, GenerateSubexperiments

class TestGenerateSubexperiments(AllenNlpTestCase):

    def setUp(self):
        super(TestGenerateSubexperiments, self).setUp()
        # not a valid experiment config. used it to for brevity of tests.
        self.root_config = Params({
                "model": {
                        "pretrained_file": "glove.6B.100d.txt.gz",
                        "embedding_dim": 100,
                        "seq2vec_encoder": {
                                "bidirectional": False,
                                "input_size": 100,
                                "hidden_size": 100
                        },
                        "classifier": {
                                "input_dim": 100, # 100 is incorrect input_dim
                                "num_layers": 2,
                                "hidden_dims": [200, 3]
                        }
                }})

        self.change_config_1 = Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                       "value_tuple": ["'glove.6B.200d.txt.gz'", "200"],
                                       "key_name": "glove_embedding_dim",
                                       "value_name": "200"})

        self.change_config_2 = Params({"key_tuple": ["model.seq2vec_encoder.bidirectional"],
                                       "value_tuple": ["!current.model.seq2vec_encoder.bidirectional"],
                                       "key_name": "bidirectional",
                                       "value_name": "toggle"})

        self.change_config_3 = Params({"key_tuple": ["model.classifier_feedforward.input_dim"],
                                       "value_tuple": [
                                               """if current.model.seq2vec_encoder.bidirectional
                                               then current.model.seq2vec_encoder.hidden_size * 2
                                               else current.model.seq2vec_encoder.hidden_size
                                               """],
                                       "key_name": "adjust",
                                       "value_name": "ff_input_dim"})

        self.group_of_changes_config_1 = Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                                 "value_tuples": [
                                                         ["'glove.6B.100d.txt.gz'", "100"],
                                                         ["'glove.6B.200d.txt.gz'", "200"],
                                                         ["'glove.6B.300d.txt.gz'", "300"]
                                                 ],
                                                 "value_names": ["100d", "200d", "300d"],
                                                 "key_name": "glove_dim"})

        self.group_of_changes_config_2 = Params({"key_tuple": ["model.seq2vec_encoder.bidirectional"],
                                                 "value_tuples": [
                                                         ["true"],
                                                         ["false"]
                                                 ], # value_names can be inferred implicitly
                                                 "key_name": "bidirectional"})

    def test_change(self):
        # test simple change (no current referecing required)
        # 1. Use 200d glove instead of 100d:
        key_tuple = ("model.pretrained_file", "model.embedding_dim")
        value_tuple = ("'glove.6B.200d.txt.gz'", "200")
        key_name = "glove_embedding_dim"
        value_name = "200"
        change1 = Change(key_tuple=key_tuple, value_tuple=value_tuple,
                         key_name=key_name, value_name=value_name)
        assert str(change1) == "glove_embedding_dim=200"
        updated_config = change1.execute(self.root_config)
        updated_config_dict = updated_config.as_dict()
        expected_updated_config_dict = deepcopy(self.root_config)
        expected_updated_config_dict["model"]["pretrained_file"] = "glove.6B.200d.txt.gz"
        expected_updated_config_dict["model"]["embedding_dim"] = 200
        assert updated_config_dict == expected_updated_config_dict

        # test complex change with jsonnet expression ('current' referece required)
        # 2. Toggle bidirectional seq2vec encoder:
        key_tuple = ("model.seq2vec_encoder.bidirectional",)
        value_tuple = ("!current.model.seq2vec_encoder.bidirectional",)
        key_name = "bidirectional"
        value_name = "toggle"
        change2 = Change(key_tuple=key_tuple, value_tuple=value_tuple,
                         key_name=key_name, value_name=value_name)
        assert str(change2) == "bidirectional=toggle"
        updated_config = change2.execute(updated_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["seq2vec_encoder"]["bidirectional"]

        # test complex change with jsonnet expression ('current' referece required)
        # 3. Reset input_dim of classifier  based on bidirectional setting
        key_tuple = ("model.classifier_feedforward.input_dim",)
        value_tuple = ("if current.model.seq2vec_encoder.bidirectional "
                       "then current.model.seq2vec_encoder.hidden_size * 2 "
                       "else current.model.seq2vec_encoder.hidden_size",)
        key_name = "adjust"
        value_name = "ff_input_dim"
        change3 = Change(key_tuple=key_tuple, value_tuple=value_tuple,
                         key_name=key_name, value_name=value_name)
        assert str(change3) == "adjust=ff_input_dim"
        updated_config = change3.execute(updated_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["classifier_feedforward"]["input_dim"] == 200

        # test default / inferred change names
        key_tuple = ("model.pretrained_file", "model.embedding_dim")
        value_tuple = ("'glove.6B.200d.txt.gz'", "200")
        change = Change(key_tuple=key_tuple, value_tuple=value_tuple)
        assert str(change) == "model.pretrained_file_model.embedding_dim='glove.6B.200d.txt.gz'_200"


    def test_change_from_params(self):
        # test simple change (no current referecing required)
        # 1. Use 200d glove instead of 100d:
        change_params = Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                "value_tuple": ["'glove.6B.200d.txt.gz'", "200"],
                                "value_name": "glove_embedding_dim",
                                "key_name": "200"})
        change1 = Change.from_params(change_params)
        updated_config = change1.execute(self.root_config)
        updated_config_dict = updated_config.as_dict()
        expected_updated_config_dict = deepcopy(self.root_config)
        expected_updated_config_dict["model"]["pretrained_file"] = "glove.6B.200d.txt.gz"
        expected_updated_config_dict["model"]["embedding_dim"] = 200
        assert updated_config_dict == expected_updated_config_dict

        # test complicated change (current object referecing required)
        # 2. Toggle bidirectional seq2vec encoder:
        change_params = Params({"key_tuple": ["model.seq2vec_encoder.bidirectional"],
                                "value_tuple": ["!current.model.seq2vec_encoder.bidirectional"]})
        change2 = Change.from_params(change_params)
        updated_config = change2.execute(updated_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["seq2vec_encoder"]["bidirectional"]

        # test complicated change (current object referecing required)
        # 3. Reset input_dim of classifier  based on bidirectional setting
        change_params = Params({"key_tuple": ["model.classifier_feedforward.input_dim"],
                                "value_tuple": [
                                        """if current.model.seq2vec_encoder.bidirectional
                                        then current.model.seq2vec_encoder.hidden_size * 2
                                        else current.model.seq2vec_encoder.hidden_size
                                    """],
                                "value_name": "adjust",
                                "key_name": "ff_input_dim"})
        change = Change.from_params(change_params)
        updated_config = change.execute(updated_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["classifier_feedforward"]["input_dim"] == 200

    def test_invalid_change(self):
        # 1. Err if key_tuple is not tuple:
        with pytest.raises(ConfigurationError):
            Change(key_tuple="model.pretrained_file",
                   value_tuple=("'glove.6B.200d.txt.gz'", "200"))
        # 2. Err if value_tuple is not tuple:
        with pytest.raises(ConfigurationError):
            Change(key_tuple=("model.pretrained_file", "model.embedding_dim"),
                   value_tuple="'glove.6B.200d.txt.gz'")
        # 3. Err if key_tuple and value_tuple and not of same size.
        with pytest.raises(ConfigurationError):
            Change(key_tuple=("model.pretrained_file", "model.embedding_dim"),
                   value_tuple=("'glove.6B.200d.txt.gz'", "200", "something"))

    def test_invalid_change_from_params(self):
        # 1. Err if key_tuple is not tuple:
        with pytest.raises(ConfigurationError):
            Change.from_params(Params({"key_tuple": "model.pretrained_file",
                                       "value_tuple": ["'glove.6B.200d.txt.gz'", "200"]}))
        # 2. Err if value_tuple is not tuple:
        with pytest.raises(ConfigurationError):
            Change.from_params(Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                       "value_tuple": "'glove.6B.200d.txt.gz'"}))
        # 3. Err if key_tuple and value_tuple and not of same size.
        with pytest.raises(ConfigurationError):
            Change.from_params(Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                       "value_tuple": ["'glove.6B.200d.txt.gz'", "200", "something"]}))

    def test_sequence_of_changes(self):
        # 1. Test a Sequence of Changes (from constructor)
        change_1 = Change.from_params(self.change_config_1)
        change_2 = Change.from_params(self.change_config_2)
        change_3 = Change.from_params(self.change_config_3)
        changes = SequenceOfChanges([change_1, change_2, change_3])
        assert str(changes) == "glove_embedding_dim=200,bidirectional=toggle,adjust=ff_input_dim"
        updated_config = changes.execute(self.root_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["pretrained_file"] == "glove.6B.200d.txt.gz"
        assert updated_config_dict["model"]["embedding_dim"] == 200
        assert updated_config_dict["model"]["seq2vec_encoder"]["bidirectional"]
        assert updated_config_dict["model"]["classifier_feedforward"]["input_dim"] == 200

        # 2. Test a Sequence of Changes (from constructor)
        # test same set of changes in different order: bidirectional toggle before last
        # ensure 'current' variable reference works appropriately
        changes = SequenceOfChanges([change_1, change_3, change_2])
        assert str(changes) == "glove_embedding_dim=200,adjust=ff_input_dim,bidirectional=toggle"
        updated_config = changes.execute(self.root_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["pretrained_file"] == "glove.6B.200d.txt.gz"
        assert updated_config_dict["model"]["embedding_dim"] == 200
        assert updated_config_dict["model"]["seq2vec_encoder"]["bidirectional"]
        assert updated_config_dict["model"]["classifier_feedforward"]["input_dim"] == 100 # Diff than 200 above

    def test_sequence_of_changes_from_params(self):
        params_list = [Params({"key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                               "value_tuple": ["'glove.6B.200d.txt.gz'", "200"]}),
                       Params({"key_tuple": ["model.seq2vec_encoder.bidirectional"],
                               "value_tuple": ["!current.model.seq2vec_encoder.bidirectional"]})]
        changes = SequenceOfChanges.from_params_list(params_list)
        updated_config = changes.execute(self.root_config)
        updated_config_dict = updated_config.as_dict()
        assert updated_config_dict["model"]["pretrained_file"] == "glove.6B.200d.txt.gz"
        assert updated_config_dict["model"]["embedding_dim"] == 200
        assert updated_config_dict["model"]["seq2vec_encoder"]["bidirectional"]

    def test_group_of_changes(self):
        # test valid group_of_changes
        change1 = Change(key_tuple=("model.pretrained_file", "model.embedding_dim"),
                         value_tuple=("'glove.6B.100d.txt.gz'", "100"))
        change2 = Change(key_tuple=("model.pretrained_file", "model.embedding_dim"),
                         value_tuple=("'glove.6B.200d.txt.gz'", "200"))
        change3 = Change(key_tuple=("model.pretrained_file", "model.embedding_dim"),
                         value_tuple=("'glove.6B.300d.txt.gz'", "300"))
        changes = GroupOfChanges([change1, change2, change3])

        assert len(changes.changes) == 3
        # test invalid group_of_changes
        change_different_key = Change(key_tuple=("model.seq2vec_encoder.bidirectional",),
                                      value_tuple=("true",))
        with pytest.raises(ConfigurationError):
            GroupOfChanges([change1, change2, change_different_key])

    def test_group_of_changes_from_params(self):
        group_of_changes_1 = GroupOfChanges.from_params(Params({
                "key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                "value_tuples": [
                        ["'glove.6B.100d.txt.gz'", "100"],
                        ["'glove.6B.200d.txt.gz'", "200"],
                        ["'glove.6B.300d.txt.gz'", "300"]
                ],
                "value_names": ["100d", "200d", "300d"],
                "key_name": "glove_dim"
                }))
        group_of_changes_1 = GroupOfChanges.from_params(self.group_of_changes_config_1) # remove above?

        group_of_changes_2 = GroupOfChanges.from_params(Params({
                "key_tuple": ["model.seq2vec_encoder.bidirectional"],
                "value_tuples": [
                        ["true"],
                        ["false"]
                ], # value_names can be inferred implicitly
                "key_name": "bidirectional"
                }))
        group_of_changes_2 = GroupOfChanges.from_params(self.group_of_changes_config_2) # remove above?

        assert len(group_of_changes_1.changes) == 3
        assert len(group_of_changes_2.changes) == 2


    def test_combination_of_groups_of_changes(self):
        group_of_changes_1 = GroupOfChanges.from_params(self.group_of_changes_config_1)
        group_of_changes_2 = GroupOfChanges.from_params(self.group_of_changes_config_2)

        to_be_combined_changes = [group_of_changes_1, group_of_changes_2]
        generator = SubExperimentsGenerator(root_config=self.root_config,
                                            to_be_combined_changes=to_be_combined_changes)

        subexperiments = generator._subexperiments
        subexperiment_configs = subexperiments.values()
        assert len(subexperiments) == 6
        # generator._subexperiments is already set, calling .save will suffice.
        # Following is only to test methods separately.

        combinations = generator._generate_combinations_of_changes()
        assert len(list(combinations)) == 6

        # Test there are 6 unique combinations
        unique_result_dicts = []
        for subexperiment_config in subexperiment_configs:
            result_dict = subexperiment_config.as_dict()
            if result_dict not in unique_result_dicts:
                unique_result_dicts.append(result_dict)
        assert len(unique_result_dicts) == 6

        # verify there are 3 and 2 values for change 1 and chage 2 respectively
        pretrained_files = {result_dict["model"]["pretrained_file"]
                            for result_dict in unique_result_dicts}
        expected_pretrained_files = {"glove.6B.100d.txt.gz",
                                     "glove.6B.200d.txt.gz",
                                     "glove.6B.300d.txt.gz"}
        assert pretrained_files == expected_pretrained_files # change 1

        embedding_dims = {result_dict["model"]["embedding_dim"]
                          for result_dict in unique_result_dicts}
        expected_embedding_dims = {100, 200, 300}
        assert embedding_dims == expected_embedding_dims # change 1

        bidirectionals = {result_dict["model"]["seq2vec_encoder"]["bidirectional"]
                          for result_dict in unique_result_dicts}
        expected_bidirectionals = {True, False}
        assert bidirectionals == expected_bidirectionals # change 2

        # Make sure these 6 combinations are different in only 2 change fields that we asked for
        unique_result_dicts = []
        for subexperiment_config in subexperiment_configs:
            result_dict = subexperiment_config.as_dict()
            result_dict["model"].pop("pretrained_file") # change1
            result_dict["model"].pop("embedding_dim")   # change1
            result_dict["model"]["seq2vec_encoder"].pop("bidirectional")   # change2
            if result_dict not in unique_result_dicts:
                unique_result_dicts.append(result_dict)
        assert len(unique_result_dicts) == 1

    def test_subexperiment_generator(self):
        generator_config = Params({
                # Make combinations of 3 different embeddings sizes and bidirectional On/Off
                "combine_changes":[
                        {
                                "key_tuple": ["model.pretrained_file", "model.embedding_dim"],
                                "value_tuples": [
                                        ["'glove.6B.100d.txt.gz'", "100"],
                                        ["'glove.6B.200d.txt.gz'", "200"],
                                        ["'glove.6B.300d.txt.gz'", "300"]
                                ],
                                "value_names": ["100d", "200d", "300d"],
                                "key_name": "glove_dim"
                        },
                        {
                                "key_tuple": ["model.seq2vec_encoder.bidirectional"],
                                "value_tuples": [
                                        ["true"],
                                        ["false"]
                                ],
                                "key_name": "bidirectional"
                        }
                ],
                # Adjust classifier input after generating each combination.
                "post_combine_changes": [
                        {
                                "key_tuple": ["model.classifier_feedforward.input_dim"],
                                "value_tuple": [
                                        """if current.model.seq2vec_encoder.bidirectional
                                        then current.model.seq2vec_encoder.hidden_size * 2
                                        else current.model.seq2vec_encoder.hidden_size
                                        """],
                                # following names won't be added in experiment id. No identifier
                                # needed because it's a change that all combinations will have.
                                "value_name": "adjust",
                                "key_name": "ff_input_dim"
                        }
                ]
        })

        # 1. Test SubExperimentGenerator.from_params
        generator = SubExperimentsGenerator.from_params(self.root_config, generator_config)
        assert len(generator._subexperiments) == 6

        expected_sub_experiment_names = {'glove_dim=100d,bidirectional=true',
                                         'glove_dim=100d,bidirectional=false',
                                         'glove_dim=200d,bidirectional=true',
                                         'glove_dim=200d,bidirectional=false',
                                         'glove_dim=300d,bidirectional=true',
                                         'glove_dim=300d,bidirectional=false'}
        assert set(generator._subexperiments.keys()) == expected_sub_experiment_names

        # test few of the generated configs:
        # a.
        subexperiment_config = generator._subexperiments['glove_dim=100d,bidirectional=true']
        assert subexperiment_config.get("model").get("pretrained_file") == "glove.6B.100d.txt.gz"
        assert subexperiment_config.get("model").get("embedding_dim") == 100
        assert subexperiment_config.get("model").get("seq2vec_encoder").get("bidirectional")

        # b.
        subexperiment_config = generator._subexperiments['glove_dim=200d,bidirectional=false']
        assert subexperiment_config.get("model").get("pretrained_file") == "glove.6B.200d.txt.gz"
        assert subexperiment_config.get("model").get("embedding_dim") == 200
        assert not subexperiment_config.get("model").get("seq2vec_encoder").get("bidirectional")

        # c.
        subexperiment_config = generator._subexperiments['glove_dim=300d,bidirectional=true']
        assert subexperiment_config.get("model").get("pretrained_file") == "glove.6B.300d.txt.gz"
        assert subexperiment_config.get("model").get("embedding_dim") == 300
        assert subexperiment_config.get("model").get("seq2vec_encoder").get("bidirectional")

        # make sure post combine adjustment changes have been applied in each.
        for _, subexperiment_config in generator._subexperiments.items():
            classifier_input = subexperiment_config.get("model").get("classifier_feedforward").get("input_dim")
            seq2vec_hidden = subexperiment_config.get("model").get("seq2vec_encoder").get("hidden_size")
            seq2vec_bidirectional = subexperiment_config.get("model").get("seq2vec_encoder").get("bidirectional")
            assert classifier_input == 2 * seq2vec_hidden if seq2vec_bidirectional else seq2vec_hidden

        # 2. Test SubExperimentGenerator.save
        subexperiments_dir = self.TEST_DIR / 'subexperiments_save'
        generator.save(subexperiments_dir, "root_exp_name")
        file_names = set(os.listdir(subexperiments_dir))

        expected_file_names = {'1.root_exp_name.glove_dim=100d,bidirectional=true.json',
                               '2.root_exp_name.glove_dim=100d,bidirectional=false.json',
                               '3.root_exp_name.glove_dim=200d,bidirectional=true.json',
                               '4.root_exp_name.glove_dim=200d,bidirectional=false.json',
                               '5.root_exp_name.glove_dim=300d,bidirectional=true.json',
                               '6.root_exp_name.glove_dim=300d,bidirectional=false.json'}
        assert expected_file_names == file_names

    def test_generate_subexperiments_args(self):
        parser = argparse.ArgumentParser(description="Testing")
        subparsers = parser.add_subparsers(title='Commands', metavar='')
        GenerateSubexperiments().add_subparser('generate-subexperiments', subparsers)

        raw_args = ["generate-subexperiments", "path/to/root_config",
                    "path/to/subexperiment_config", "path/to/output_dir"]
        args = parser.parse_args(raw_args)

        assert args.func == generate_subexperiments_from_args
        assert args.root_experiment_file == "path/to/root_config"
        assert args.generator_file == "path/to/subexperiment_config"
        assert args.subexperiments_dir == "path/to/output_dir"

        # all three arguments are required and no more permitted
        with self.assertRaises(SystemExit) as cm: # pylint: disable=invalid-name
            args = parser.parse_args(["generate-subexperiments", "path_arg1"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        with self.assertRaises(SystemExit) as cm: # pylint: disable=invalid-name
            args = parser.parse_args(["generate-subexperiments", "path_arg1", "path_arg2"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        with self.assertRaises(SystemExit) as cm: # pylint: disable=invalid-name
            args = parser.parse_args(["generate-subexperiments", "path_arg1", "path_arg2",
                                      "path_arg3", "path_arg4"])
            assert cm.exception.code == 2  # argparse code for incorrect usage
