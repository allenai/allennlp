import pytest
import torch

from allennlp.common.params import Params
from allennlp.modules.span_extractors import SpanExtractor
from allennlp.modules.span_extractors.max_pooling_span_extractor import MaxPoolingSpanExtractor


class TestMaxPoolingSpanExtractor:
    def test_locally_span_extractor_can_build_from_params(self):
        params = Params(
            {
                "type": "max_pooling",
                "input_dim": 3,
                "num_width_embeddings": 5,
                "span_width_embedding_dim": 3,
            }
        )
        extractor = SpanExtractor.from_params(params)
        assert isinstance(extractor, MaxPoolingSpanExtractor)
        assert extractor.get_output_dim() == 6

    def test_max_values_extracted(self):
        # Test if max_pooling is correctly applied
        # We use a high dimensional random vector and assume that a randomly correct result is too unlikely
        sequence_tensor = torch.randn([2, 10, 30])
        extractor = MaxPoolingSpanExtractor(30)

        indices = torch.LongTensor([[[1, 1], [2, 4], [9, 9]], [[0, 1], [4, 4], [0, 9]]])
        span_representations = extractor(sequence_tensor, indices)

        assert list(span_representations.size()) == [2, 3, 30]
        assert extractor.get_output_dim() == 30
        assert extractor.get_input_dim() == 30

        # We iterate over the tensor to compare the span extractors's results
        # with the results of python max operation over each dimension for each span and for each batch
        # For each batch
        for batch, X in enumerate(indices):
            # For each defined span index
            for indices_ind, span_def in enumerate(X):

                # original features of current tested span
                # span_width x embedding dim (30)
                span_features_complete = sequence_tensor[batch][span_def[0] : span_def[1] + 1]

                # comparison for each dimension
                for i in range(extractor.get_output_dim()):
                    # get the features for dimension i of current span
                    features_from_span = span_features_complete[:, i]
                    real_max_value = max(features_from_span)

                    extracted_max_value = span_representations[batch, indices_ind, i]

                    assert real_max_value == extracted_max_value, (
                        f"Error extracting max value for "
                        f"batch {batch}, span {indices_ind} on dimension {i}."
                        f"expected {real_max_value} "
                        f"but got {extracted_max_value} which is "
                        f"not the maximum element."
                    )

    def test_sequence_mask_correct_excluded(self):
        # Check if span indices masked out by the sequence mask are ignored when computing
        # the span representations. For this test span_start is valid, but span_end is masked out.

        sequence_tensor = torch.randn([2, 6, 30])

        extractor = MaxPoolingSpanExtractor(30)
        indices = torch.LongTensor([[[1, 1], [3, 5], [2, 5]], [[0, 0], [0, 3], [4, 5]]])
        # define sequence mak
        seq_mask = torch.BoolTensor([[True] * 4 + [False] * 2, [True] * 5 + [False] * 1])

        span_representations = extractor(sequence_tensor, indices, sequence_mask=seq_mask)

        # After we computed the representations we set values to -inf
        # to compute the "real" max-pooling with python's max function.
        sequence_tensor[torch.logical_not(seq_mask)] = float("-inf")

        # Comparison is similar to test_max_values_extracted
        for batch, X in enumerate(indices):
            for indices_ind, span_def in enumerate(X):

                span_features_complete = sequence_tensor[batch][span_def[0] : span_def[1] + 1]

                for i, _ in enumerate(span_features_complete):
                    features_from_span = span_features_complete[:, i]
                    real_max_value = max(features_from_span)
                    extracted_max_value = span_representations[batch, indices_ind, i]

                    assert real_max_value == extracted_max_value, (
                        f"Error extracting max value for "
                        f"batch {batch}, span {indices_ind} on dimension {i}."
                        f"expected {real_max_value} "
                        f"but got {extracted_max_value} which is "
                        f"not the maximum element."
                    )

    def test_span_mask_correct_excluded(self):
        # All masked out span indices by span_mask should be '0'

        sequence_tensor = torch.randn([2, 6, 10])

        extractor = MaxPoolingSpanExtractor(10)
        indices = torch.LongTensor([[[1, 1], [3, 5], [2, 5]], [[0, 0], [0, 3], [4, 5]]])

        span_mask = torch.BoolTensor([[True] * 3, [False] * 3])

        span_representations = extractor(
            sequence_tensor,
            indices,
            span_indices_mask=span_mask,
        )

        # The span-mask masks out all indices in the last batch
        # We check whether all span representations for this batch are '0'
        X = indices[-1]
        batch = -1
        for indices_ind, span_def in enumerate(X):

            span_features_complete = sequence_tensor[batch][span_def[0] : span_def[1] + 1]

            for i, _ in enumerate(span_features_complete):
                real_max_value = torch.FloatTensor([0.0])
                extracted_max_value = span_representations[batch, indices_ind, i]

                assert real_max_value == extracted_max_value, (
                    f"Error extracting max value for "
                    f"batch {batch}, span {indices_ind} on dimension {i}."
                    f"expected {real_max_value} "
                    f"but got {extracted_max_value} which is "
                    f"not the maximum element."
                )

    def test_inconsistent_extractor_dimension_throws_exception(self):

        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[1, 1], [2, 4], [9, 9]], [[0, 1], [4, 4], [0, 9]]])

        with pytest.raises(ValueError):
            extractor = MaxPoolingSpanExtractor(9)
            extractor(sequence_tensor, indices)

        with pytest.raises(ValueError):
            extractor = MaxPoolingSpanExtractor(11)
            extractor(sequence_tensor, indices)

    def test_span_indices_outside_sequence(self):

        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[6, 6], [2, 4]], [[0, 1], [4, 4]]])

        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

        indices = torch.LongTensor([[[5, 6], [2, 4]], [[0, 1], [4, 4]]])

        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

        indices = torch.LongTensor([[[-1, 0], [2, 4]], [[0, 1], [4, 4]]])

        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

    def test_span_start_below_span_end(self):

        sequence_tensor = torch.randn([2, 6, 10])
        indices = torch.LongTensor([[[4, 2], [2, 4], [1, 1]], [[0, 1], [4, 4], [1, 1]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices)

    def test_span_sequence_complete_masked(self):

        sequence_tensor = torch.randn([2, 6, 10])
        seq_mask = torch.BoolTensor([[True] * 2 + [False] * 4, [True] * 3 + [False] * 3])
        indices = torch.LongTensor([[[5, 5]], [[4, 5]]])
        with pytest.raises(IndexError):
            extractor = MaxPoolingSpanExtractor(10)
            extractor(sequence_tensor, indices, sequence_mask=seq_mask)
