# pylint: disable=no-self-use,invalid-name
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import AtisDatasetReader
from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.worlds import AtisWorld

class TestAtisReader(AllenNlpTestCase):
    def test_atis_read_from_file(self):
        data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "atis" / "sample.json"
        database_file = cached_path("https://s3-us-west-2.amazonaws.com/allennlp/datasets/atis/atis.db")
        reader = AtisDatasetReader(database_file=database_file)

        instances = list(reader.read(str(data_path)))

        assert len(instances) == 13
        instance = instances[0]

        assert set(instance.fields.keys()) == \
                {'utterance',
                 'actions',
                 'world',
                 'sql_queries',
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

        assert world.linked_entities['string']['airport_airport_code_string -> ["\'DTW\'"]'][2] == \
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # ``detroit`` -> ``DTW``
        assert world.linked_entities['string']['flight_stop_stop_airport_string -> ["\'DTW\'"]'][2] == \
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # ``detroit`` -> ``DTW``
        assert world.linked_entities['string']['city_city_code_string -> ["\'DDTT\'"]'][2] == \
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # ``detroit`` -> ``DDTT``
        assert world.linked_entities['string']['fare_basis_economy_string -> ["\'NO\'"]'][2] == \
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0] # ``one way`` -> ``NO``
        assert world.linked_entities['string']['city_city_name_string -> ["\'WESTCHESTER COUNTY\'"]'][2] == \
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] # ``westchester county`` -> ``WESTCHESTER COUNTY``
        assert world.linked_entities['string']['city_city_code_string -> ["\'HHPN\'"]'][2] == \
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1] # ``westchester county`` -> ``HHPN``
