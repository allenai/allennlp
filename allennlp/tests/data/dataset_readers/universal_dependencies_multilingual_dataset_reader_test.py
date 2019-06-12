# pylint: disable=no-self-use,invalid-name

from allennlp.data.dataset_readers import UniversalDependenciesMultiLangDatasetReader
from allennlp.common.testing import AllenNlpTestCase

class TestUniversalDependenciesMultilangDatasetReader(AllenNlpTestCase):
    data_path = AllenNlpTestCase.FIXTURES_ROOT / "data" / "dependencies_multilang" / "*"

    def check_two_instances(self, instance1, instance2):
        fields1, fields2 = instance1.fields, instance2.fields
        assert fields1['metadata'].metadata['lang'] == fields2['metadata'].metadata['lang']

        lang = fields1['metadata'].metadata['lang']
        if lang == 'fr':
            assert fields1['metadata'].metadata['lang'] == 'fr'
            assert [t.text for t in fields1["words"].tokens] == ['Ses', 'habitants', 'sont', 'appelés', 'les',
                                                                 'Paydrets',
                                                                 'et', 'les', 'Paydrètes', ';']
            assert fields1["pos_tags"].labels == ['DET', 'NOUN', 'VERB', 'VERB', 'DET',
                                                  'NOUN', 'CONJ', 'DET', 'NOUN', '.']
            assert fields1["head_tags"].labels == ['poss', 'nsubjpass', 'auxpass', 'ROOT', 'det', 'attr',
                                                   'cc', 'det', 'conj', 'p']
            assert fields1["head_indices"].labels == [2, 4, 4, 0, 6, 4, 6, 9, 6, 4]

            assert fields2['metadata'].metadata['lang'] == 'fr'
            assert [t.text for t in fields2["words"].tokens] == ['Cette', 'tour', 'de', 'a',
                                                                 'été', 'achevée', 'en', '1962', '.']
            assert fields2["pos_tags"].labels == ['DET', 'NOUN', 'ADP', 'VERB', 'VERB',
                                                  'VERB', 'ADP', 'NUM', '.']
            assert fields2["head_tags"].labels == ['det', 'nsubjpass', 'adpmod', 'aux', 'auxpass', 'ROOT',
                                                   'adpmod', 'adpobj', 'p']
            assert fields2["head_indices"].labels == [2, 6, 2, 6, 6, 0, 6, 7, 6]

        elif lang == 'es':
            assert [t.text for t in fields1["words"].tokens] == ['Aclarando', 'hacia', 'todo', 'el', 'mundo',
                                                                 'Valderrama', 'Y', 'Eduardo', 'Son', 'La',
                                                                 'Misma', 'Persona', '.']

            assert fields1["pos_tags"].labels == ['VERB', 'ADP', 'DET', 'DET', 'NOUN', 'NOUN', 'CONJ',
                                                  'NOUN', 'NOUN', 'DET', 'ADJ', 'NOUN', '.']
            assert fields1["head_tags"].labels == ['ROOT', 'adpmod', 'det', 'det', 'adpobj', 'nsubj', 'cc', 'conj',
                                                   'xcomp',
                                                   'det', 'amod', 'attr', 'p']
            assert fields1["head_indices"].labels == [0, 1, 5, 5, 2, 9, 6, 6, 1, 12, 12, 9, 1]

            assert [t.text for t in fields2["words"].tokens] == ['Es', 'un', 'bar', 'disfrazado', 'de',
                                                                 'restaurante', 'la', 'comida', 'esta',
                                                                 'demasiado', 'salada', '.']
            assert fields2["pos_tags"].labels == ['VERB', 'DET', 'NOUN', 'VERB', 'ADP', 'NOUN',
                                                  'DET', 'NOUN', 'VERB', 'PRON', 'ADJ', '.']
            assert fields2["head_tags"].labels == ['ROOT', 'det', 'attr', 'partmod', 'adpmod', 'adpobj',
                                                   'det', 'nsubj', 'parataxis', 'nmod', 'acomp', 'p']
            assert fields2["head_indices"].labels == [0, 3, 1, 3, 4, 5, 8, 9, 1, 11, 9, 1]

        elif lang == 'it':
            assert fields1['metadata'].metadata['lang'] == 'it'
            assert [t.text for t in fields1["words"].tokens] == ['Inconsueto', 'allarme', 'alla', 'Tate',
                                                                 'Gallery', ':']
            assert fields1["pos_tags"].labels == ['ADJ', 'NOUN', 'ADP', 'NOUN', 'NOUN', '.']
            assert fields1["head_tags"].labels == ['amod', 'ROOT', 'adpmod', 'dep', 'adpobj', 'p']
            assert fields1["head_indices"].labels == [2, 0, 2, 5, 3, 2]

            assert fields2['metadata'].metadata['lang'] == 'it'
            assert [t.text for t in fields2["words"].tokens] == ['Hamad', 'Butt', 'è', 'morto', 'nel', '1994',
                                                                 'a', '32', 'anni', '.']
            assert fields2["pos_tags"].labels == ['NOUN', 'NOUN', 'VERB', 'VERB', 'ADP',
                                                  'NUM', 'ADP', 'NUM', 'NOUN', '.']
            assert fields2["head_tags"].labels == ['dep', 'nsubj', 'aux', 'ROOT', 'adpmod', 'adpobj',
                                                   'adpmod', 'num', 'adpobj', 'p']
            assert fields2["head_indices"].labels == [2, 4, 4, 0, 4, 5, 4, 9, 7, 4]

        return lang

    def test_iterate_once_per_file_when_first_pass_for_vocab_is_true(self):
        reader = UniversalDependenciesMultiLangDatasetReader(
                languages=['es', 'fr', 'it'], is_first_pass_for_vocab=True)
        instances = list(reader.read(str(self.data_path)))
        assert len(instances) == 6

        processed_langs = []

        processed_langs.append(self.check_two_instances(instances[0], instances[1]))
        processed_langs.append(self.check_two_instances(instances[2], instances[3]))
        processed_langs.append(self.check_two_instances(instances[4], instances[5]))

        assert 'es' in processed_langs and 'fr' in processed_langs and 'it' in processed_langs

    def test_iterate_forever_when_first_pass_for_vocab_is_false(self):
        '''
        Note: assumes that each data file contains no more than 20 trees.
        '''
        reader = UniversalDependenciesMultiLangDatasetReader(languages=['es', 'fr', 'it'],
                                                             is_first_pass_for_vocab=False,
                                                             instances_per_file=1,
                                                             lazy=True)
        counter_es, counter_fr, counter_it = 0, 0, 0
        for instance in reader.read(str(self.data_path)):
            lang = instance.fields['metadata'].metadata['lang']
            if lang == 'es':
                counter_es += 1
                if counter_es > 20:
                    break
            if lang == 'fr':
                counter_fr += 1
                if counter_fr > 20:
                    break
            if lang == 'it':
                counter_it += 1
                if counter_it > 20:
                    break
        # Asserting that the reader didn't stop after reading the three files once.
        assert (counter_es > 20 or counter_fr > 20 or counter_it > 20)
