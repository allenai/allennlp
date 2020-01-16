from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers.spacy_indexer import SpacyTokenIndexer
from allennlp.data.fields.text_field import TextField
from allennlp.common.util import get_spacy_model
from allennlp.data.vocabulary import Vocabulary


class TestSpacyTokenIndexer(AllenNlpTestCase):
    def test_as_array_produces_token_array(self):
        indexer = SpacyTokenIndexer()
        nlp = get_spacy_model("en_core_web_sm", pos_tags=True, parse=False, ner=False)
        tokens = [t for t in nlp("This is a sentence.")]
        field = TextField(tokens, token_indexers={"spacy": indexer})

        vocab = Vocabulary()
        field.index(vocab)

        # Indexer functionality
        array_dict = indexer.tokens_to_indices(tokens, vocab)
        assert len(array_dict["tokens"]) == 5
        assert len(array_dict["tokens"][0]) == 96

        # Check it also works with field
        lengths = field.get_padding_lengths()
        array_dict = field.as_tensor(lengths)

        assert list(array_dict["spacy"]["tokens"].shape) == [5, 96]
