# pylint: disable=no-self-use,invalid-name
import itertools
import math

from pytest import approx
from flaky import flaky
import torch
from torch.autograd import Variable

from allennlp.modules import ConditionalRandomField
from allennlp.common.testing import AllenNlpTestCase


class TestConditionalRandomField(AllenNlpTestCase):
    @flaky
    def test_forward_works_without_mask(self):
        START_TAG = 0
        END_TAG = 1

        logits = Variable(torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ]))
        tags = Variable(torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ]))

        # Use the CRF Module to compute the log_likelihood
        crf = ConditionalRandomField(5, START_TAG, END_TAG)
        log_likelihood = crf.forward(logits, tags).data[0]

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        def score(logits, tags):
            """
            Computes the likelihood score for the given sequence of tags,
            given the provided logits (and the transition weights in the CRF model)
            """
            # Start with transitions from START and to END
            total = crf.transitions[tags[0], START_TAG] + crf.transitions[END_TAG, tags[-1]]
            # Add in all the intermediate transitions
            for tag, next_tag in zip(tags, tags[1:]):
                total += crf.transitions[next_tag, tag]
            # Add in the logits for the observed tags
            for logit, tag in zip(logits, tags):
                total += logit[tag]
            return total.data[0]

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in zip(logits, tags):
            numerator = score(logits_i.data, tags_i.data)
            all_scores = [score(logits_i.data, tags_j) for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood == approx(log_likelihood)


    def test_forward_works_with_mask(self):
        logits = Variable(torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ]))
        labels = Variable(torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ]))
        mask = Variable(torch.ByteTensor([
                [1, 1, 1],
                [1, 1, 0]
        ]))

        ConditionalRandomField(5, 0, 1).forward(logits, labels, mask)
