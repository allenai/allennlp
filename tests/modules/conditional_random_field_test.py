# pylint: disable=no-self-use,invalid-name
import itertools
import math

from pytest import approx, raises
import torch
from torch.autograd import Variable

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase


class TestConditionalRandomField(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.logits = Variable(torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ]))
        self.tags = Variable(torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ]))

        self.transitions = torch.Tensor([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = ConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    def test_forward_works_without_mask(self):
        log_likelihood = self.crf(self.logits, self.tags).data[0]

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(self.logits, self.tags):
            numerator = self.score(logits_i.data, tags_i.data)
            all_scores = [self.score(logits_i.data, tags_j) for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood == approx(log_likelihood)


    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        mask = Variable(torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ]))

        log_likelihood = self.crf(self.logits, self.tags, mask).data[0]

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in zip(self.logits, self.tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.data)
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood == approx(log_likelihood)


    def test_viterbi_tags(self):
        mask = Variable(torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ]))

        viterbi_tags = self.crf.viterbi_tags(self.logits, mask)

        # Check that the viterbi tags are what I think they should be.
        assert viterbi_tags == [
                [2, 4, 3],
                [4, 2]
        ]

        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []

        for logit, mas in zip(self.logits, mask):
            sequence_length = torch.sum(mas.data)
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))

        assert viterbi_tags == most_likely_tags

    def test_constrained_viterbi_tags(self):
        constraints = {(0, 0), (0, 1),
                       (1, 1), (1, 2),
                       (2, 2), (2, 3),
                       (3, 3), (3, 4),
                       (4, 4), (4, 0)}

        crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        crf.transitions = torch.nn.Parameter(self.transitions)
        crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        mask = Variable(torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ]))

        viterbi_tags = crf.viterbi_tags(self.logits, mask)

        # Now the tags should respect the constraints
        assert viterbi_tags == [
                [2, 3, 3],
                [2, 3]
        ]

    def test_allowed_transitions(self):
        # pylint: disable=bad-whitespace,bad-continuation
        bio_labels = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']
        #              0     1      2      3      4
        allowed = allowed_transitions("BIO", dict(enumerate(bio_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == {
            (0, 0), (0, 1),         (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (2, 0), (2, 1), (2, 2), (2, 3),
            (3, 0), (3, 1),         (3, 3), (3, 4),
            (4, 0), (4, 1),         (4, 3), (4, 4)
        }

        bioul_labels = ['O', 'B-X', 'I-X', 'L-X', 'U-X', 'B-Y', 'I-Y', 'L-Y', 'U-Y']
        #                0     1      2      3      4      5      6      7      8
        allowed = allowed_transitions("BIOUL", dict(enumerate(bioul_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == {
            (0, 0), (0, 1),                 (0, 4), (0, 5),                 (0, 8),
                            (1, 2), (1, 3),
                            (2, 2), (2, 3),
            (3, 0), (3, 1),                 (3, 4), (3, 5),                 (3, 8),
            (4, 0), (4, 1),                 (4, 4), (4, 5),                 (4, 8),
                                                            (5, 6), (5, 7),
                                                            (6, 6), (6, 7),
            (7, 0), (7, 1),                 (7, 4), (7, 5),                 (7, 8),
            (8, 0), (8, 1),                 (8, 4), (8, 5),                 (8, 8)
        }

        with raises(ConfigurationError):
            allowed_transitions("allennlp", {})
