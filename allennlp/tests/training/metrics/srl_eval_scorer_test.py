# pylint: disable=no-self-use,invalid-name,protected-access
from numpy.testing import assert_allclose

from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.metrics import SrlEvalScorer


class SrlEvalScorerTest(AllenNlpTestCase):
    def test_srl_eval_correctly_scores_identical_tags(self):
        batch_verb_indices = [3, 8, 2]
        batch_sentences = [["Mali", "government", "officials", "say", "the", "woman", "'s",
                            "confession", "was", "forced", "."],
                           ["Mali", "government", "officials", "say", "the", "woman", "'s",
                            "confession", "was", "forced", "."],
                           ['The', 'prosecution', 'rested', 'its', 'case', 'last', 'month', 'after',
                            'four', 'months', 'of', 'hearings', '.']]
        batch_predicted_tags = [['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                                 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O'],
                                ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1',
                                 'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O'],
                                ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
                                 'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                                 'I-ARGM-TMP', 'I-ARGM-TMP', 'O']]
        batch_gold_tags = [['B-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'B-ARG1',
                            'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O'],
                           ['O', 'O', 'O', 'O', 'B-ARG1', 'I-ARG1',
                            'I-ARG1', 'I-ARG1', 'B-V', 'B-ARG2', 'O'],
                           ['B-ARG0', 'I-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP',
                            'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP',
                            'I-ARGM-TMP', 'I-ARGM-TMP', 'O']]

        srl_scorer = SrlEvalScorer()
        srl_scorer(batch_verb_indices, batch_sentences, batch_predicted_tags, batch_gold_tags)
        metrics = srl_scorer.get_metric()
        for metric in metrics.values():
            assert_allclose(metric, 1.0)
