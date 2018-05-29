# pylint: disable=no-self-use
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import AgendaPredictionDatasetReader


class TestAgendaPredictionDatasetReader(AllenNlpTestCase):
    def test_reader_reads(self):
        test_file = "tests/fixtures/data/nlvr/sample_processed_data.jsonl"
        dataset = AgendaPredictionDatasetReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 2
        instance = instances[0]
        assert instance.fields.keys() == {"sentence", "all_actions", "target_actions"}
        sentence_tokens = instance.fields["sentence"].tokens
        expected_tokens = ["There", "is", "a", "circle", "closely", "touching", "a", "corner", "of",
                           "a", "box", "."]
        assert [t.text for t in sentence_tokens] == expected_tokens
        all_actions = [action.rule for action in instance.fields["all_actions"].field_list]
        target_action_labels = instance.fields["target_actions"].field_list
        assert len(all_actions) == len(target_action_labels)
        in_actions = set()
        for target_action_label, action in zip(target_action_labels, all_actions):
            if target_action_label.label == "in":
                in_actions.add(action)
        assert in_actions == {'<o,o> -> circle', '<o,o> -> touch_corner', '<o,o> -> touch_wall',
                              '<o,t> -> object_exists', 'o -> all_objects'}
