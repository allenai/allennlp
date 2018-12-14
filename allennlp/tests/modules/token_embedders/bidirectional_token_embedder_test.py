# pylint: disable=no-self-use,invalid-name
import filecmp
import json
import os
import pathlib
import tarfile

import torch

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.modules.token_embedders import ElmoTokenEmbedder


class TestElmoTokenEmbedder(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'bidirectional_lm' / 'characters_token_embedder.json',
                          self.FIXTURES_ROOT / 'data' / 'conll2003.txt')

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_elmo_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict['tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                assert tag in {'O', 'I-ORG', 'I-PER', 'I-LOC'}
