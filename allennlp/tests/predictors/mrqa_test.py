# pylint: disable=no-self-use,invalid-name
from pytest import approx

from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.reading_comprehension.mrqa_reader import *
from allennlp.models.reading_comprehension.BERT_QA import *
from allennlp.predictors.mrqa_predictor import *
import numpy as np

class TestMCQAPredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
      inputs = {
        "answerKey": "B",
        "id": "70701f5d1d62e58d5c74e2e303bb4065",
        "question": {
          "choices": [
            {
              "label": "A",
              "text": "building"
            },
            {
              "label": "B",
              "text": "table"
            },
            {
              "label": "C",
              "text": "curtains"
            },
            {
              "label": "D",
              "text": "-"
            },
            {
              "label": "E",
              "text": "-"
            }
          ],
          "stem": "A look out form my window, what do a see? "
        }
      }

      file_path = cached_path('s3://multiqa/models/CSQA_BERT/base.tar.gz')
      archive = load_archive(file_path)
      predictor = Predictor.from_archive(archive, 'bert_mc_qa')

      result = predictor.predict_json(inputs)

      labels = {0:'A',1:'B',2:'C',3:'D',4:'E'}

      print('\n\n the predicated label is %s' % (labels[np.argmax(result['instance']['label_probs'])]))


