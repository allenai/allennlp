# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import AtisDatasetReader 
from allennlp.common.testing import AllenNlpTestCase 


class TestAtisReader(AllenNlpTestCase):

    def test_atis_read_from_file(self):
        reader = AtisDatasetReader()
        data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "atis" / "sample.json"

        instances = list(reader.read(str(data_path)))

        instance = instances[0]
        valid_strs = set()
        for prod in instance.fields['actions'].field_list:
            if prod.rule.startswith('string'):
                valid_strs.add(prod.rule)
        assert valid_strs == set(instance.fields['world'].metadata['string'])
        
        instance = instances[1]
        valid_strs = set()
        for prod in instance.fields['actions'].field_list:
            if prod.rule.startswith('string'):
                valid_strs.add(prod.rule)

        assert valid_strs == set(instance.fields['world'].metadata['string'])

