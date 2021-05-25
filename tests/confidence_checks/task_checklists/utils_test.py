from allennlp.confidence_checks.task_checklists import utils
from allennlp.common.testing import AllenNlpTestCase


class TestUtils(AllenNlpTestCase):
    def test_punctuations(self):
        perturbed = utils.toggle_punctuation("This has a period.")

        assert perturbed[0] == "This has a period"

        perturbed = utils.toggle_punctuation("This does not have a period")
        assert perturbed[0] == "This does not have a period."
