import numpy
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.fields import TextField, SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class TestSpanField(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.indexers = {"words": SingleIdTokenIndexer("words")}
        self.text = TextField(
            [Token(t) for t in ["here", "is", "a", "sentence", "for", "spans", "."]], self.indexers
        )

    def test_as_tensor_converts_span_field_correctly(self):
        span_field = SpanField(2, 3, self.text)
        tensor = span_field.as_tensor(span_field.get_padding_lengths()).detach().cpu().numpy()
        numpy.testing.assert_array_equal(tensor, numpy.array([2, 3]))

    def test_span_field_raises_on_incorrect_label_type(self):
        with pytest.raises(TypeError):
            _ = SpanField("hello", 3, self.text)

    def test_span_field_raises_on_ill_defined_span(self):
        with pytest.raises(ValueError):
            _ = SpanField(4, 1, self.text)

    def test_span_field_raises_if_span_end_is_greater_than_sentence_length(self):
        with pytest.raises(ValueError):
            _ = SpanField(1, 30, self.text)

    def test_empty_span_field_works(self):
        span_field = SpanField(1, 3, self.text)
        empty_span = span_field.empty_field()
        assert empty_span.span_start == -1
        assert empty_span.span_end == -1

    def test_printing_doesnt_crash(self):
        span_field = SpanField(2, 3, self.text)
        print(span_field)

    def test_equality(self):
        span_field1 = SpanField(2, 3, self.text)
        span_field2 = SpanField(2, 3, self.text)
        span_field3 = SpanField(
            2, 3, TextField([Token(t) for t in ["not", "the", "same", "tokens"]], self.indexers)
        )

        assert span_field1 == (2, 3)
        assert span_field1 == span_field1
        assert span_field1 == span_field2
        assert span_field1 != span_field3
        assert span_field2 != span_field3
