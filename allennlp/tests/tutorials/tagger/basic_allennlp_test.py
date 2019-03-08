from allennlp.common.testing import AllenNlpTestCase


class TestBasicAllenNlp(AllenNlpTestCase):
    @classmethod
    def test_run_as_script(cls):
        # Just ensure the tutorial runs without throwing an exception.
        import tutorials.tagger.basic_allennlp # pylint: disable=unused-variable
