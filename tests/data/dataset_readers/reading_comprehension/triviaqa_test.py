# pylint: disable=no-self-use,invalid-name
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import TriviaQaReader


class TestTriviaQaReader(AllenNlpTestCase):
    def test_read(self):
        params = Params({
                'base_tarball_path': 'tests/fixtures/data/triviaqa-sample.tgz',
                })
        reader = TriviaQaReader.from_params(params)
        instances = reader.read('web-train.json').instances
        assert len(instances) == 3

        assert [t.text for t in instances[0].fields["question"].tokens[:3]] == ["Which", "American", "-"]
        assert [t.text for t in instances[0].fields["passage"].tokens[:3]] == ["The", "Nobel", "Prize"]
        url = "http://www.nobelprize.org/nobel_prizes/literature/laureates/1930/"
        assert [t.text for t in instances[0].fields["passage"].tokens[-3:]] == ["<", url, ">"]
        assert instances[0].fields["span_start"].sequence_index == 12
        assert instances[0].fields["span_end"].sequence_index == 13

        assert [t.text for t in instances[1].fields["question"].tokens[:3]] == ["Which", "American", "-"]
        assert [t.text for t in instances[1].fields["passage"].tokens[:3]] == ["Why", "Do", "n’t"]
        assert [t.text for t in instances[1].fields["passage"].tokens[-3:]] == ["adults", ",", "and"]
        assert instances[1].fields["span_start"].sequence_index == 38
        assert instances[1].fields["span_end"].sequence_index == 39

        assert [t.text for t in instances[2].fields["question"].tokens[:3]] == ["Where", "in", "England"]
        assert [t.text for t in instances[2].fields["passage"].tokens[:3]] == ["Judi", "Dench", "-"]
        assert [t.text for t in instances[2].fields["passage"].tokens[-3:]] == [")", "(", "special"]
        assert instances[2].fields["span_start"].sequence_index == 16
        assert instances[2].fields["span_end"].sequence_index == 16
