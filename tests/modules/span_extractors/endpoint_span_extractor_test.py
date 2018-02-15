# pylint: disable=no-self-use,invalid-name
import numpy
import torch
from torch.autograd import Variable

from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.common.params import Params
from allennlp.nn.util import batched_index_select

class TestEndpointSpanExtractor:
    def test_endpoint_span_extractor_can_build_from_params(self):
        params = Params({
                "type": "endpoint",
                "input_dim": 7,
                "num_width_embeddings": 5,
                "span_width_embedding_dim": 3
                })
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, EndpointSpanExtractor)
        assert extractor.get_output_dim() == 10

    def test_correct_sequence_elements_are_embedded(self):
        sequence_tensor = Variable(torch.randn([2, 5, 7]))
        # concatentate start and end points together to form our representation.
        extractor = EndpointSpanExtractor(7, "x,y")

        indices = Variable(torch.LongTensor([[[1, 3],
                                              [2, 4]],
                                             [[0, 2],
                                              [3, 4]]]))
        span_representations = extractor(sequence_tensor, indices)

        assert list(span_representations.size()) == [2, 2, 14]
        assert extractor.get_output_dim() == 14
        assert extractor.get_input_dim() == 7

        start_indices, end_indices = indices.split(1, -1)
        # We just concatenated the start and end embeddings together, so
        # we can check they match the original indices if we split them apart.
        start_embeddings, end_embeddings = span_representations.split(7, -1)

        correct_start_embeddings = batched_index_select(sequence_tensor, start_indices.squeeze())
        correct_end_embeddings = batched_index_select(sequence_tensor, end_indices.squeeze())
        numpy.testing.assert_array_equal(start_embeddings.data.numpy(),
                                         correct_start_embeddings.data.numpy())
        numpy.testing.assert_array_equal(end_embeddings.data.numpy(),
                                         correct_end_embeddings.data.numpy())

    def test_masked_indices_are_handled_correctly(self):
        sequence_tensor = Variable(torch.randn([2, 5, 7]))
        # concatentate start and end points together to form our representation.
        extractor = EndpointSpanExtractor(7, "x,y")

        indices = Variable(torch.LongTensor([[[1, 3],
                                              [2, 4]],
                                             [[0, 2],
                                              [3, 4]]]))
        span_representations = extractor(sequence_tensor, indices)

        # Make a mask with the second batch element completely masked.
        indices_mask = Variable(torch.LongTensor([[1, 1], [0, 0]]))

        span_representations = extractor(sequence_tensor, indices, span_indices_mask=indices_mask)
        start_embeddings, end_embeddings = span_representations.split(7, -1)
        start_indices, end_indices = indices.split(1, -1)

        correct_start_embeddings = batched_index_select(sequence_tensor, start_indices.squeeze()).data
        # Completely masked second batch element, so it should all be zero.
        correct_start_embeddings[1, :, :].fill_(0)
        correct_end_embeddings = batched_index_select(sequence_tensor, end_indices.squeeze()).data
        correct_end_embeddings[1, :, :].fill_(0)
        numpy.testing.assert_array_equal(start_embeddings.data.numpy(),
                                         correct_start_embeddings.numpy())
        numpy.testing.assert_array_equal(end_embeddings.data.numpy(),
                                         correct_end_embeddings.numpy())
