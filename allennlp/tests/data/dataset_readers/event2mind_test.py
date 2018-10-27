# pylint: disable=no-self-use,invalid-name
from typing import cast

import pytest

from allennlp.data.dataset_readers import Event2MindDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

def get_text(key: str, instance: Instance):
    return [t.text for t in cast(TextField, instance.fields[key]).tokens]

class TestEvent2MindDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read(self, lazy):
        reader = Event2MindDatasetReader(lazy=lazy)
        instances = reader.read(
                str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'event2mind_small.csv')
        )
        instances = ensure_list(instances)

        assert len(instances) == 13
        instance = instances[0]
        assert get_text("source", instance) == ["@start@", "it", "is", "personx", "'s",
                                                "favorite", "animal", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "none", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "excited", "to", "see", "it", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "none", "@end@"]

        instance = instances[3]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "move", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "grateful", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "charitable", "@end@"]

        instance = instances[4]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "move", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "grateful", "@end@"]
        # Interestingly, taking all combinations doesn't make much sense if the
        # original source is ambiguous.
        assert get_text("oreact", instance) == ["@start@", "enraged", "@end@"]

        instance = instances[10]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "steal", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "guilty", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "enraged", "@end@"]

        instance = instances[11]
        assert get_text("source", instance) == ["@start@", "personx", "gets", "persony",
                                                "'s", "mother", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "helpful", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "useful", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "grateful", "@end@"]

        instance = instances[12]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "for", "fun", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "happy", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "like", "a", "good", "friend", "@end@"]

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_with_dummy_instances_for_vocab_generation(self, lazy):
        reader = Event2MindDatasetReader(lazy=lazy, dummy_instances_for_vocab_generation=True)
        instances = reader.read(
                str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'event2mind_small.csv')
        )
        instances = ensure_list(instances)

        assert len(instances) == 17
        instance = instances[0]
        assert get_text("source", instance) == ["@start@", "it", "is", "personx", "'s",
                                                "favorite", "animal", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "none", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "none", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "none", "@end@"]

        instance = instances[5]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "move", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "none", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "none", "@end@"]

        instance = instances[7]
        assert get_text("source", instance) == ["@start@", "none", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "none", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "grateful", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "none", "@end@"]

        instance = instances[9]
        assert get_text("source", instance) == ["@start@", "none", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "none", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "none", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "charitable", "@end@"]

        instance = instances[14]
        assert get_text("source", instance) == ["@start@", "personx", "drives",
                                                "persony", "'s", "truck", "@end@"]
        assert get_text("xintent", instance) == ["@start@", "for", "fun", "@end@"]
        assert get_text("xreact", instance) == ["@start@", "none", "@end@"]
        assert get_text("oreact", instance) == ["@start@", "none", "@end@"]
