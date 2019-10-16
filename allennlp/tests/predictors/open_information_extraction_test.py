from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.predictors.open_information_extraction import (
    consolidate_predictions,
    get_predicate_text,
)
from allennlp.predictors.open_information_extraction import sanitize_label
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


class TestOpenIePredictor(AllenNlpTestCase):
    def test_uses_named_inputs(self):
        """
        Tests whether the model outputs conform to the expected format.
        """
        inputs = {
            "sentence": "Angela Merkel met and spoke to her EU counterparts during the climate summit."
        }

        archive = load_archive(self.FIXTURES_ROOT / "srl" / "serialization" / "model.tar.gz")
        predictor = Predictor.from_archive(archive, "open-information-extraction")

        result = predictor.predict_json(inputs)

        words = result.get("words")
        assert words == [
            "Angela",
            "Merkel",
            "met",
            "and",
            "spoke",
            "to",
            "her",
            "EU",
            "counterparts",
            "during",
            "the",
            "climate",
            "summit",
            ".",
        ]
        num_words = len(words)

        verbs = result.get("verbs")
        assert verbs is not None
        assert isinstance(verbs, list)

        for verb in verbs:
            tags = verb.get("tags")
            assert tags is not None
            assert isinstance(tags, list)
            assert all(isinstance(tag, str) for tag in tags)
            assert len(tags) == num_words

    def test_sanitize_label(self):

        assert sanitize_label("B-ARGV-MOD") == "B-ARGV-MOD"

    def test_prediction_with_no_verbs(self):
        """
        Tests whether the model copes with sentences without verbs.
        """
        input1 = {"sentence": "Blah no verb sentence."}
        archive = load_archive(self.FIXTURES_ROOT / "srl" / "serialization" / "model.tar.gz")
        predictor = Predictor.from_archive(archive, "open-information-extraction")

        result = predictor.predict_json(input1)
        assert result == {"words": ["Blah", "no", "verb", "sentence", "."], "verbs": []}

    def test_predicate_consolidation(self):
        """
        Test whether the predictor can correctly consolidate multiword
        predicates.
        """
        tokenizer = SpacyTokenizer(pos_tags=True)

        sent_tokens = tokenizer.tokenize("In December, John decided to join the party.")

        # Emulate predications - for both "decided" and "join"
        predictions = [
            ["B-ARG2", "I-ARG2", "O", "B-ARG0", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O"],
            ["O", "O", "O", "B-ARG0", "B-BV", "I-BV", "B-V", "B-ARG1", "I-ARG1", "O"],
        ]
        # Consolidate
        pred_dict = consolidate_predictions(predictions, sent_tokens)

        # Check that only "decided to join" is left
        assert len(pred_dict) == 1
        tags = list(pred_dict.values())[0]
        assert get_predicate_text(sent_tokens, tags) == "decided to join"

    def test_more_than_two_overlapping_predicates(self):
        """
        Test whether the predictor can correctly consolidate multiword
        predicates.
        """
        tokenizer = SpacyTokenizer(pos_tags=True)

        sent_tokens = tokenizer.tokenize("John refused to consider joining the club.")

        # Emulate predications - for "refused" and "consider" and "joining"
        predictions = [
            ["B-ARG0", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "I-ARG1", "O"],
            ["B-ARG0", "B-BV", "I-BV", "B-V", "B-ARG1", "I-ARG1", "I-ARG1", "O"],
            ["B-ARG0", "B-BV", "I-BV", "I-BV", "B-V", "B-ARG1", "I-ARG1", "O"],
        ]

        # Consolidate
        pred_dict = consolidate_predictions(predictions, sent_tokens)

        # Check that only "refused to consider to join" is left
        assert len(pred_dict) == 1
        tags = list(pred_dict.values())[0]
        assert get_predicate_text(sent_tokens, tags) == "refused to consider joining"
