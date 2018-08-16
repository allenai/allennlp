# pylint: disable=no-self-use,invalid-name
from allennlp.data.dataset_readers import AtisDatasetReader
from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.worlds import AtisWorld

class TestAtisReader(AllenNlpTestCase):
    def test_atis_read_from_file(self):
        reader = AtisDatasetReader()
        data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "atis" / "sample.json"

        instances = list(reader.read(str(data_path)))
        
        assert len(instances) == 11
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
        # Check that the strings in the actions field has the same actions
        # as the strings in the world.
        valid_strs = set()
        for action in instance.fields['actions'].field_list:
            if action.rule.startswith('string'):
                valid_strs.add(action.rule)

        world = instance.fields['world'].metadata
        assert valid_strs == set(world.valid_actions['string'])

        # We make sure all the entities are sorted in reverse order because
        # we need to enforce some ordering for it to match the correct row
        # in the linking scores.
        assert sorted(world.valid_actions['string'], reverse=True) == \
                world.valid_actions['string']

        assert sorted(world.valid_actions['number'], reverse=True) == \
                world.valid_actions['number']

        assert world.valid_actions['string'] == \
            ['string -> ["\'WESTCHESTER COUNTY\'"]',
             'string -> ["\'NO\'"]',
             'string -> ["\'HHPN\'"]',
             'string -> ["\'DTW\'"]',
             'string -> ["\'DETROIT\'"]',
             'string -> ["\'DDTT\'"]']

        assert world.valid_actions['number'] == \
                ['number -> ["1"]',
                 'number -> ["0"]']

        # We should have generated created linking scores of the shape
        # (number_entities, number_utterance_tokens). We have two types
        # of entities: strings and numbers.
        assert world.linking_scores.shape[0] == \
                len(world.valid_actions['string'] +
                    world.valid_actions['number'])
        assert world.linking_scores.shape[1] == \
                len(instance.fields['utterance'].tokens)
