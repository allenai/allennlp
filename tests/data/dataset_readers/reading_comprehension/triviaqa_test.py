# pylint: disable=no-self-use,invalid-name
import pytest
import numpy as np

from allennlp.common import Params
from allennlp.common.util import ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import TriviaQaReader
from allennlp.data.vocabulary import Vocabulary

def get_params(data_format, **kwargs):
    if data_format == "tar.gz":
        params = Params({
                'triviaqa_path': 'tests/fixtures/data/triviaqa-sample.tgz',
                'paragraph_picker': 'triviaqa-web-train',
                'sample_first_iteration': True,
                'lazy': True
        })
    elif data_format == "preprocessed":
        params = Params({
                'triviaqa_path': 'tests/fixtures/data/triviaqa_processed',
                'paragraph_picker': 'triviaqa-web-train',
                'sample_first_iteration': True,
                'data_format': 'preprocessed',
                'lazy': True
        })
    else:
        raise ValueError(f"unknown format {data_format}")

    for k, v in kwargs.items():
        params[k] = v

    return params

class TestTriviaQaReader:
    @pytest.mark.parametrize('paragraph_picker', ('triviaqa-web-train', None))
    def test_sampling(self, paragraph_picker):
        params = get_params("tar.gz", paragraph_picker=paragraph_picker)
        reader = TriviaQaReader.from_params(params)

        # Read the dataset 10 times, if we're supposed to sample, we should get
        # varying results. If we're not supposed to sample, we shouldn't.
        first_words = set()
        for _ in range(10):
            instances = ensure_list(reader.read('web-train.json'))
            instance = instances[0]
            paragraphs = instance.fields['paragraphs']
            key = ';'.join([
                    paragraphs[0].tokens[0].text,
                    paragraphs[0].tokens[-1].text,
                    paragraphs[1].tokens[0].text,
                    paragraphs[1].tokens[-1].text,
            ])
            first_words.add(key)

        if paragraph_picker == "triviaqa-web-train":
            # Sampling should get us different results.
            assert len(first_words) > 1
        else:
            # Should not sample, so only one result.
            assert len(first_words) == 1

    @pytest.mark.parametrize('max_token_length', (4, 8))
    def test_max_token_length(self, max_token_length):
        params = get_params('tar.gz', max_token_length=max_token_length)

        reader = TriviaQaReader.from_params(params)
        instances = ensure_list(reader.read('web-train.json'))

        paragraph_max = max(len(token.text)
                            for instance in instances
                            for paragraph in instance.fields['paragraphs']
                            for token in paragraph.tokens)

        question_max = max(len(token.text)
                           for instance in instances
                           for token in instance.fields['question'].tokens)

        assert paragraph_max == max_token_length
        assert question_max == 10

    @pytest.mark.parametrize("data_format", ("tar.gz", "preprocessed"))
    def test_read(self, data_format):
        # Dataset reader samples, so let's set the seed for deterministic tests.
        np.random.seed(121)

        params = get_params(data_format)
        reader = TriviaQaReader.from_params(params)

        filename = 'web-train.jsonl' if data_format == "preprocessed" else 'web-train.json'
        instances = reader.read(filename)
        instances = ensure_list(instances)

        assert len(instances) == 3

        instance = instances[0]
        question = instance.fields['question']
        assert [t.text for t in question.tokens] == [
                "Which", "American", "-", "born", "Sinclair",
                "won", "the", "Nobel", "Prize", "for", "Literature",
                "in", "1930", "?"]

        paragraphs = instance.fields['paragraphs']
        assert len(paragraphs) == 2

        assert [t.text for t in paragraphs[0].tokens][:4] == ['a', 'boorish', 'mass', 'of']
        assert [t.text for t in paragraphs[1].tokens][:4] == ['The', 'Nobel', 'Prize', 'in']

        # ListField[ListField[SpanField]] -> List[List[Tuple[int, int]]]
        spans = [[(span.span_start, span.span_end) for span in spans]
                 for spans in instance.fields['spans']]
        assert spans[0] == [(-1, -1)]
        assert spans[1] == [(14, 15), (24, 25), (41, 42), (159, 160)]

    def test_tensors(self):
        params = get_params('tar.gz')
        reader = TriviaQaReader.from_params(params)
        instances = reader.read('web-train.json')
        instances = ensure_list(instances)
        vocab = Vocabulary.from_instances(instances)

        batch = Batch(instances)
        batch.index_instances(vocab)

        padding_lengths = batch.get_padding_lengths()
        tensors = batch.as_tensor_dict(padding_lengths)

        assert set(tensors) == {'question', 'paragraphs', 'spans', 'metadata'}
