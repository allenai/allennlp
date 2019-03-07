from allennlp.common.testing import AllenNlpTestCase

class TestBasicAllenNlp(AllenNlpTestCase):
    def test_run_as_script(self):
        import tutorials.tagger.basic_allennlp
