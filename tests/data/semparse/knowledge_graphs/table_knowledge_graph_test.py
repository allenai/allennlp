# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.semparse.knowledge_graphs import TableKnowledgeGraph


class TestTableKnowledgeGraph(AllenNlpTestCase):
    def test_read_from_json_handles_simple_cases(self):
        json = {
                'columns': ['Name in English', 'Location'],
                'cells': [['Paradeniz', 'Mersin'],
                          ['Lake Gala', 'Edirne']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_cell_neighbors('fb:cell.mersin'))
        assert neighbors == {'fb:row.row.location'}
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.name_in_english'))
        assert neighbors == {'fb:cell.paradeniz', 'fb:cell.lake_gala'}

    def test_read_from_json_handles_diacritics(self):
        json = {
                'columns': ['Name in English', 'Name in Turkish', 'Location'],
                'cells': [['Lake Van', 'Van Gölü', 'Mersin'],
                          ['Lake Gala', 'Gala Gölü', 'Edirne']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.name_in_turkish'))
        assert neighbors == {'fb:cell.van_golu', 'fb:cell.gala_golu'}

    def test_read_from_json_handles_newlines_in_columns(self):
        # The TSV file we use has newlines converted to "\n", not actual escape characters.  We
        # need to be sure we catch this.
        json = {
                'columns': ['Peak\\nAUS', 'Peak\\nNZ'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.peak_aus'))
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.peak_nz'))
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(knowledge_graph.get_cell_neighbors('fb:cell.1'))
        assert neighbors == {'fb:row.row.peak_aus'}

        json = {
                'columns': ['Title'],
                'cells': [['Dance of the\\nSeven Veils']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.title'))
        assert neighbors == {'fb:cell.dance_of_the_seven_veils'}

    def test_read_from_json_handles_diacritics_and_newlines(self):
        json = {
                'columns': ['Notes'],
                'cells': [['8 districts\nFormed from Orūzgān Province in 2004']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.notes'))
        assert neighbors == {'fb:cell.8_districts_formed_from_oruzgan_province_in_2004'}

    def test_read_from_json_handles_parentheses_correctly(self):
        json = {
                'columns': ['Urban settlements'],
                'cells': [['Dzhebariki-Khaya\\n(Джебарики-Хая)'],
                          ['South Korea (KOR)'],
                          ['Area (km²)']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row.urban_settlements'))
        assert neighbors == {'fb:cell.dzhebariki_khaya',
                             'fb:cell.south_korea_kor',
                             'fb:cell.area_km'}

    def test_read_from_json_handles_columns_with_duplicate_normalizations(self):
        json = {
                'columns': ['# of votes', '% of votes'],
                'cells': [['1', '2'],
                          ['3', '4']]
                }
        knowledge_graph = TableKnowledgeGraph.read_from_json(json)
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row._of_votes'))
        assert neighbors == {'fb:cell.1', 'fb:cell.3'}
        neighbors = set(knowledge_graph.get_column_neighbors('fb:row.row._of_votes_2'))
        assert neighbors == {'fb:cell.2', 'fb:cell.4'}
        neighbors = set(knowledge_graph.get_cell_neighbors('fb:cell.1'))
        assert neighbors == {'fb:row.row._of_votes'}
