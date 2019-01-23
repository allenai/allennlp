# pylint: disable=no-self-use,invalid-name
import numpy

from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestLanguageModelingDatasetReader:
    def test_read_from_file_no_fuzz_is_deterministic(self):
        """
        The dataset is split into 4 batches, becoming:

        [[This, is, a, sentence],
         [for, language, modelling, .],
         [</S>, Here, 's, another],
         [one, for, extra, language]]

        Then, since our truncated bptt size is 2, our the inputs of our first batch
        consists of:
        [[This, is],
         [for, language],
         [</S>, Here],
         [one, for]]

        The second batch consists of:
        [[a],
         [modelling],
         ['s],
         [extra]]

        Note that the second batch has a shorter sequence length of 1, and we do not
        predict labels for the final words in the batch.
        """
        # Results should be identical if we run twice.
        for _ in range(2):
            reader = LanguageModelingReader(batch_size=4,
                                            truncated_bptt_size=2,
                                            fuzz_truncated_bptt_size=False)
            instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'data' /
                                                'language_modeling.txt'))
            # This should match the batch size
            assert len(instances[0].fields["inputs"].field_list) == 4
            assert len(instances[0].fields["forward_targets"].field_list) == 4
            # This is the number of batches generated
            assert len(instances) == 2

            first_instance_inputs = [["This", "is"],
                                     ["for", "language"],
                                     ["</S>", "Here"],
                                     ["one", "for"]]
            first_instance_forward_targets = [["is", "a"],
                                      ["language", "modelling"],
                                      ["Here", "'s"],
                                      ["for", "extra"]]

            first_instance_generated_inputs = [
                    [x.text for x in instances[0].fields["inputs"].field_list[i].tokens] for
                    i in range(len(instances[0].fields["inputs"].field_list))]
            assert first_instance_generated_inputs == first_instance_inputs
            first_instance_generated_forward_targets = [
                    [x.text for x in instances[0].fields["forward_targets"].field_list[i].tokens] for
                    i in range(len(instances[0].fields["forward_targets"].field_list))]
            assert first_instance_generated_forward_targets == first_instance_forward_targets

            second_instance_inputs = [["a"],
                                      ["modelling"],
                                      ["'s"],
                                      ["extra"]]
            second_instance_forward_targets = [["sentence"],
                                       ["."],
                                       ["another"],
                                       ["language"]]
            second_instance_generated_inputs = [
                    [x.text for x in instances[1].fields["inputs"].field_list[i].tokens] for
                    i in range(len(instances[1].fields["inputs"].field_list))]
            assert second_instance_generated_inputs == second_instance_inputs
            second_instance_generated_forward_targets = [
                    [x.text for x in instances[1].fields["forward_targets"].field_list[i].tokens] for
                    i in range(len(instances[1].fields["forward_targets"].field_list))]
            assert second_instance_generated_forward_targets == second_instance_forward_targets

    def test_read_from_file(self):
        """
        The dataset is split into 2 batches, becoming:

        [[This, is, a, sentence, for, language, modelling, ., </S>],
         [Here, 's, another, one, for, extra, language, modelling, .]]

        Our truncated bptt size is 2, but fuzz_truncated_bptt_size is True. So
        the sequence length is randomly perturbed, becoming 5. As a result,
        the inputs are:

        [[This, is, a, sentence, for],
         [Here, 's, another, one, for]]

        The second batch consists of:
        [[language, modelling, .],
         [extra, language, modelling]]

        Note that the second batch has a shorter sequence length of 3, and we do not
        predict labels for the final words in the batch.
        """
        numpy.random.seed(seed=0)
        reader = LanguageModelingReader(batch_size=2, truncated_bptt_size=2)
        instances = ensure_list(reader.read(AllenNlpTestCase.FIXTURES_ROOT / 'data' /
                                            'language_modeling.txt'))
        # This should match the batch size
        assert len(instances[0].fields["inputs"].field_list) == 2
        # This is the number of batches generated
        assert len(instances) == 2

        first_instance_inputs = [["This", "is", "a", "sentence", "for"],
                                 ["Here", "'s", "another", "one", "for"]]
        first_instance_forward_targets = [["is", "a", "sentence", "for", "language"],
                                  ["'s", "another", "one", "for", "extra"]]
        first_instance_generated_inputs = [
                [x.text for x in instances[0].fields["inputs"].field_list[i].tokens] for
                i in range(len(instances[0].fields["inputs"].field_list))]
        assert first_instance_generated_inputs == first_instance_inputs
        first_instance_generated_forward_targets = [
                [x.text for x in instances[0].fields["forward_targets"].field_list[i].tokens] for
                i in range(len(instances[0].fields["forward_targets"].field_list))]
        assert first_instance_generated_forward_targets == first_instance_forward_targets

        second_instance_inputs = [["language", "modelling", "."],
                                  ["extra", "language", "modelling"]]
        second_instance_forward_targets = [["modelling", ".", "</S>"],
                                   ["language", "modelling", "."]]
        second_instance_generated_inputs = [
                [x.text for x in instances[1].fields["inputs"].field_list[i].tokens] for
                i in range(len(instances[1].fields["inputs"].field_list))]
        assert second_instance_generated_inputs == second_instance_inputs
        second_instance_generated_forward_targets = [
                [x.text for x in instances[1].fields["forward_targets"].field_list[i].tokens] for
                i in range(len(instances[1].fields["forward_targets"].field_list))]
        assert second_instance_generated_forward_targets == second_instance_forward_targets
