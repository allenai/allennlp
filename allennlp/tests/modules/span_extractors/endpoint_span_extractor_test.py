import numpy
import torch

from allennlp.modules.span_extractors import SpanExtractor, EndpointSpanExtractor
from allennlp.common.params import Params
from allennlp.nn.util import batched_index_select


class TestEndpointSpanExtractor:
    def test_endpoint_span_extractor_can_build_from_params(self):
        params = Params(
            {
                "type": "endpoint",
                "input_dim": 7,
                "num_width_embeddings": 5,
                "span_width_embedding_dim": 3,
            }
        )
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, EndpointSpanExtractor)
        assert extractor.get_output_dim() == 17  # 2 * input_dim + span_width_embedding_dim

    def test_correct_sequence_elements_are_embedded(self):
        sequence_tensor = torch.randn([2, 5, 7])
        # Concatentate start and end points together to form our representation.
        extractor = EndpointSpanExtractor(7, "x,y")

        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [3, 4]]])
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
        numpy.testing.assert_array_equal(
            start_embeddings.data.numpy(), correct_start_embeddings.data.numpy()
        )
        numpy.testing.assert_array_equal(
            end_embeddings.data.numpy(), correct_end_embeddings.data.numpy()
        )

    def test_masked_indices_are_handled_correctly(self):
        sequence_tensor = torch.randn([2, 5, 7])
        # concatentate start and end points together to form our representation.
        extractor = EndpointSpanExtractor(7, "x,y")

        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [3, 4]]])
        span_representations = extractor(sequence_tensor, indices)

        # Make a mask with the second batch element completely masked.
        indices_mask = torch.tensor([[True, True], [False, False]])

        span_representations = extractor(sequence_tensor, indices, span_indices_mask=indices_mask)
        start_embeddings, end_embeddings = span_representations.split(7, -1)
        start_indices, end_indices = indices.split(1, -1)

        correct_start_embeddings = batched_index_select(
            sequence_tensor, start_indices.squeeze()
        ).data
        # Completely masked second batch element, so it should all be zero.
        correct_start_embeddings[1, :, :].fill_(0)
        correct_end_embeddings = batched_index_select(sequence_tensor, end_indices.squeeze()).data
        correct_end_embeddings[1, :, :].fill_(0)
        numpy.testing.assert_array_equal(
            start_embeddings.data.numpy(), correct_start_embeddings.numpy()
        )
        numpy.testing.assert_array_equal(
            end_embeddings.data.numpy(), correct_end_embeddings.numpy()
        )

    def test_masked_indices_are_handled_correctly_with_exclusive_indices(self):
        sequence_tensor = torch.randn([2, 5, 8])
        # concatentate start and end points together to form our representation
        # for both the forward and backward directions.
        extractor = EndpointSpanExtractor(8, "x,y", use_exclusive_start_indices=True)
        indices = torch.LongTensor([[[1, 3], [2, 4]], [[0, 2], [0, 1]]])
        sequence_mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )

        span_representations = extractor(sequence_tensor, indices, sequence_mask=sequence_mask)

        # We just concatenated the start and end embeddings together, so
        # we can check they match the original indices if we split them apart.
        start_embeddings, end_embeddings = span_representations.split(8, -1)

        correct_start_indices = torch.LongTensor([[0, 1], [-1, -1]])
        # These indices should be -1, so they'll be replaced with a sentinel. Here,
        # we'll set them to a value other than -1 so we can index select the indices and
        # replace them later.
        correct_start_indices[1, 0] = 1
        correct_start_indices[1, 1] = 1

        correct_end_indices = torch.LongTensor([[3, 4], [2, 1]])

        correct_start_embeddings = batched_index_select(
            sequence_tensor.contiguous(), correct_start_indices
        )
        # This element had sequence_tensor index of 0, so it's exclusive index is the start sentinel.
        correct_start_embeddings[1, 0] = extractor._start_sentinel.data
        correct_start_embeddings[1, 1] = extractor._start_sentinel.data
        numpy.testing.assert_array_equal(
            start_embeddings.data.numpy(), correct_start_embeddings.data.numpy()
        )

        correct_end_embeddings = batched_index_select(
            sequence_tensor.contiguous(), correct_end_indices
        )
        numpy.testing.assert_array_equal(
            end_embeddings.data.numpy(), correct_end_embeddings.data.numpy()
        )
