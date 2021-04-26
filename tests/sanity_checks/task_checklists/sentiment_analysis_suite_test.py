from allennlp.sanity_checks.task_checklists.sentiment_analysis_suite import SentimentAnalysisSuite
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestSentimentAnalysisSuite(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        archive = load_archive(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        self.predictor = Predictor.from_archive(archive)

    def test_run(self):
        data = [
            "This is really good",
            "This was terrible",
            "This was not good",
            "John Smith acted very well.",
            "Seattle was very gloomy.",
            "I have visited the place for 3 years; great food!",
        ]
        suite = SentimentAnalysisSuite(add_default_tests=True, data=data)
        suite.run(self.predictor, max_examples=10)
