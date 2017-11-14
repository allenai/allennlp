# pylint: disable=invalid-name,protected-access
from allennlp.common.testing import ModelTestCase


class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super(AcademicPaperClassifierTest, self).setUp()
        self.set_up_model('tests/fixtures/academic_paper_classifier/experiment.json',
                          'tests/fixtures/data/s2_papers.jsonl')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
