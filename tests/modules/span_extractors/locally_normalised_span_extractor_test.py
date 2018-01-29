# pylint: disable=no-self-use,invalid-name, protected-access
import numpy
import torch
from torch.autograd import Variable

from allennlp.modules.span_extractors import SpanExtractor, LocallyNormalisedSpanExtractor
from allennlp.common.params import Params

class TestLocallyNormalisedSpanExtractor:
    def test_locally_normalised_span_extractor_can_build_from_params(self):
        params = Params({"type": "locally_normalised", "input_dim": 5, "max_span_width": 10})
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, LocallyNormalisedSpanExtractor)

    def test_attention_is_normalised_correctly(self):
        input_dim = 7
        sequence_tensor = Variable(torch.randn([2, 5, input_dim]))
        # concatentate start and end points together to form our representation.
        extractor = LocallyNormalisedSpanExtractor(input_dim=input_dim, max_span_width=10)

        # In order to test the attention, we'll make the weight which computes the logits
        # zero, so the attention distribution is uniform over the sentence. This lets
        # us check that the computed spans are just the averages of their representations.
        extractor._global_attention._module.weight.data.fill_(0.0)
        extractor._global_attention._module.bias.data.fill_(0.0)

        indices = Variable(torch.LongTensor([[[1, 3],
                                              [2, 4]],
                                             [[0, 2],
                                              [3, 4]]])) # smaller span tests masking.
        span_representations = extractor(sequence_tensor, indices)
        assert list(span_representations.size()) == [2, 2, input_dim]

        # First element in the batch.
        batch_element = 0
        spans = span_representations[batch_element]
        # First span.
        mean_embeddings = sequence_tensor[batch_element, 1:4, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        # Second span.
        mean_embeddings = sequence_tensor[batch_element, 2:5, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), mean_embeddings.data.numpy())
        # Now the second element in the batch.
        batch_element = 1
        spans = span_representations[batch_element]
        # First span.
        mean_embeddings = sequence_tensor[batch_element, 0:3, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[0].data.numpy(), mean_embeddings.data.numpy())
        # Second span.
        mean_embeddings = sequence_tensor[batch_element, 3:5, :].mean(0)
        numpy.testing.assert_array_almost_equal(spans[1].data.numpy(), mean_embeddings.data.numpy())
