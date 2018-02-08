# pylint: disable=no-self-use,invalid-name
from flaky import flaky
import pytest
import numpy
from numpy.testing import assert_almost_equal
import torch
from torch.autograd import Variable

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import BidirectionalAttentionFlow, Model


class MultiParagraphReadingComprehensionTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            'tests/fixtures/multi_paragraph/experiment.json',
            'web-train.json'
        )
        # 'tests/fixtures/data/triviaqa-sample.tgz

    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()

        # This is a temporary hack. Right now the triviaqa dataset reader creates an Instance
        # for each (question, paragraph) pair. For multiparagraph question answering we want
        # Instances of (question, [paragraph1, paragraph2, ...]). Until I make the dataset reader
        # do that, I'm using `unsqueeze` to fake it.
        training_tensors['span_start'] = training_tensors['span_start'].unsqueeze(1)
        training_tensors['span_end'] = training_tensors['span_end'].unsqueeze(1)
        for k, v in training_tensors['passage'].items():
            training_tensors['passage'][k] = v.unsqueeze(1)

        output_dict = self.model(**training_tensors)
        # metrics = self.model.get_metrics(reset=True)

        assert 'span_start_logits' in output_dict
