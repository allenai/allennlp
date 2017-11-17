from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.wikitables import TableKnowledgeGraph
from allennlp.data.fields import KnowledgeGraphField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.common.testing import AllenNlpTestCase


class TestKnowledgeGraphField(AllenNlpTestCase):
    def test_field_returns_arrays_correctly(self):
        table_filename = "tests/fixtures/data/wikitables/sample_table.tsv"
        table_kg = TableKnowledgeGraph.read_from_file(table_filename)
        table_kg_field = KnowledgeGraphField(table_kg,
                                             token_indexers={"entity_name": SingleIdTokenIndexer("names"),
                                                             "entity_type": SingleIdTokenIndexer("types")})
        vocab = Vocabulary()
        table_kg_field.index(vocab)
        print(table_kg_field._indexed_entities)
        padding_lengths = table_kg_field.get_padding_lengths()
        print(padding_lengths)
        print(table_kg_field.as_array(padding_lengths))
