# pylint: disable=no-self-use,invalid-name
import pytest
import numpy as np

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import TriviaQaReader
from allennlp.data.vocabulary import Vocabulary


class TestTriviaQaReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        # Dataset reader samples, so let's set the seed for deterministic tests.
        np.random.seed(121)

        params = Params({
                'base_tarball_path': 'tests/fixtures/data/triviaqa-sample.tgz',
                'paragraph_picker': 'triviaqa-web-train',
                'lazy': lazy
                })
        reader = TriviaQaReader.from_params(params)

        instances = reader.read('web-train.json')
        instances = ensure_list(instances)

        assert len(instances) == 5

        instance = instances[0]
        question = instance.fields['question']
        paragraphs = instance.fields['paragraphs']
        spans = [[(span.span_start, span.span_end) for span in paragraph_spans]
                 for paragraph_spans in instance.fields['spans']]
        assert [t.text for t in question.tokens[:3]] == ["Which", "American", "-"]
        assert len(paragraphs) == 2
        assert [token.text for token in paragraphs[0]] == ['Sinclair', 'Lewis']
        assert spans == [[(0, 1)], [(-1, -1)]]

        instance = instances[1]
        question = instance.fields['question']
        paragraphs = instance.fields['paragraphs']
        spans = [[(span.span_start, span.span_end) for span in paragraph_spans]
                 for paragraph_spans in instance.fields['spans']]
        assert [t.text for t in question.tokens[:3]] == ["Which", "American", "-"]
        assert len(paragraphs) == 2
        assert spans == [[(149, 150)], [(-1, -1)]]
        assert [token.text for token in paragraphs[0].tokens[149:151]] == ['Sinclair', 'Lewis']

        # instance = instances[2]
        # question = instance.fields['question']
        # paragraphs = instance.fields['paragraphs']
        # spans = [[(span.span_start, span.span_end) for span in paragraph_spans]
        #          for paragraph_spans in instance.fields['spans']]
        # assert [t.text for t in question.tokens[:3]] == ["Which", "American", "-"]
        # assert len(paragraphs) == 2
        # assert spans == [[(149, 150)], [(-1, -1)]]
        # assert [token.text for token in paragraphs[0].tokens[149:151]] == ['Sinclair', 'Lewis']

    def test_tensors(self):
        params = Params({
                'base_tarball_path': 'tests/fixtures/data/triviaqa-sample.tgz',
                'paragraph_picker': 'triviaqa-web-train',
                'lazy': False
        })
        reader = TriviaQaReader.from_params(params)
        instances = reader.read('web-train.json')
        instances = ensure_list(instances)
        vocab = Vocabulary.from_instances(instances)

        batch = Batch(instances)
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensors = batch.as_tensor_dict(padding_lengths)
