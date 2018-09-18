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
        
        assert len(instances) == 13
        instance = instances[0]
        
        assert set(instance.fields.keys()) == \
                {'utterance',
                 'actions',
                 'world',
                 'example_sql_query',
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
        
        # TODO test the linking scores
