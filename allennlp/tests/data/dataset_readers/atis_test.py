# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import AtisDatasetReader
from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.worlds import AtisWorld

class TestAtisReader(AllenNlpTestCase):
    def test_atis_read_from_file(self):
        data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "atis" / "sample.json"
        database_directory = AllenNlpTestCase.FIXTURES_ROOT / "data" / "atis" / "atis.db"
        reader = AtisDatasetReader(database_directory=str(database_directory))

        instances = list(reader.read(str(data_path)))

        assert len(instances) == 14
        instance = instances[0]

        assert instance.fields.keys() == \
                {'utterance',
                 'actions',
                 'world',
                 'target_action_sequence',
                 'linking_scores'}

        assert [t.text for t in instance.fields["utterance"].tokens] == \
                ['show', 'me', 'the', 'one', 'way',
                 'flights', 'from', 'detroit', 'to',
                 'westchester', 'county']

        assert isinstance(instance.fields['world'].as_tensor({}), AtisWorld)

        world = instance.fields['world'].metadata
        assert world.valid_actions['number'] == \
                ['number -> ["1"]',
                 'number -> ["0"]']

        # We should have generated created linking scores of the shape
        # (num_entities, num_utterance_tokens). We have two types
        # of entities: strings and numbers.
        assert world.linking_scores.shape[0] == \
                len(world.valid_actions['number'])
        assert world.linking_scores.shape[1] == \
                len(instance.fields['utterance'].tokens)
