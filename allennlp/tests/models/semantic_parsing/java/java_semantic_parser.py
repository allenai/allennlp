# pylint: disable=invalid-name,no-self-use,protected-access
import pytest

from allennlp.common.testing import ModelTestCase

@pytest.mark.java
class JavaSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(JavaSemanticParserTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "java" / "experiment.json"),
                          str(self.FIXTURES_ROOT / "data" / "java" / "sample_data_prototypes.java"))

    def tearDown(self):
        super().tearDown()

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)