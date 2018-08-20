# pylint: disable=no-self-use,invalid-name
import pytest

from typing import cast

from allennlp.data.dataset_readers import Event2MindDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

def get_text(key: str, instance: Instance):
    return [t.text for t in cast(TextField, instance.fields[key]).tokens]

#Source,Event,Xintent,Xemotion,Otheremotion,Xsent,Osent
#it_events,It is PersonX's favorite animal,"[""none""]","[""excited to see it"", ""happy"", ""lucky""]","[""none""]",,4.0
#rocstory,PersonX drives PersonY's truck,"[""to move"", ""to steal""]","[""grateful"", ""guilty""]","[""charitable"", ""enraged""]",3.0,5.0
#rocstory,PersonX gets PersonY's mother,"[""to be helpful""]","[""useful""]","[""grateful""]",3.0,4.0

class TestEvent2MindDatasetReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_default_format(self, lazy):
        reader = Event2MindDatasetReader(lazy=lazy)
        instances = reader.read(str(AllenNlpTestCase.FIXTURES_ROOT / 'data' / 'event2mind.csv'))
        instances = ensure_list(instances)

        assert len(instances) == 12
        instance = instances[0]
        assert get_text("source_tokens",instance) == ["@start@", "it", "is", "personx", "'s",
                                                      "favorite", "animal", "@end@"]
        assert get_text("xintent_tokens",instance) == ["@start@", "none", "@end@"]
        assert get_text("xreact_tokens",instance) == ["@start@", "excited", "to", "see", "it", "@end@"]
        assert get_text("oreact_tokens",instance) == ["@start@", "none", "@end@"]

        instance = instances[3]
        assert get_text("source_tokens",instance) == ["@start@", "personx", "drives",
                                                      "persony", "'s", "truck", "@end@"]
        assert get_text("xintent_tokens",instance) == ["@start@", "move", "@end@"]
        assert get_text("xreact_tokens",instance) == ["@start@", "grateful", "@end@"]
        assert get_text("oreact_tokens",instance) == ["@start@", "charitable", "@end@"]

        instance = instances[4]
        assert get_text("source_tokens",instance) == ["@start@", "personx", "drives",
                                                      "persony", "'s", "truck", "@end@"]
        assert get_text("xintent_tokens",instance) == ["@start@", "move", "@end@"]
        assert get_text("xreact_tokens",instance) == ["@start@", "grateful", "@end@"]
        # Interestingly, taking all combinations doesn't make much sense if the original source is
        # ambiguous.
        assert get_text("oreact_tokens",instance) == ["@start@", "enraged", "@end@"]

        instance = instances[10]
        assert get_text("source_tokens",instance) == ["@start@", "personx", "drives",
                                                      "persony", "'s", "truck", "@end@"]
        assert get_text("xintent_tokens",instance) == ["@start@", "steal", "@end@"]
        assert get_text("xreact_tokens",instance) == ["@start@", "guilty", "@end@"]
        assert get_text("oreact_tokens",instance) == ["@start@", "enraged", "@end@"]

        instance = instances[11]
        assert get_text("source_tokens",instance) == ["@start@", "personx", "gets", "persony",
                                        "'s", "mother", "@end@"]
        assert get_text("xintent_tokens",instance) == ["@start@", "helpful", "@end@"]
        assert get_text("xreact_tokens",instance) == ["@start@", "useful", "@end@"]
        assert get_text("oreact_tokens",instance) == ["@start@", "grateful", "@end@"]
