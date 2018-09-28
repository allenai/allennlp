import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.bidirectional_lm import BidirectionalLanguageModel
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.softmax import Softmax
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.nn.util import get_text_field_mask


class TestBidirectionalLM(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        sentences = ["This is the first sentence.", "This is yet another sentence."]
        token_indexers = {
                "tokens": SingleIdTokenIndexer(),
                "token_characters": TokenCharactersIndexer()
        }
        tokenizer = WordTokenizer()

        self.instances = [Instance({"tokens": TextField(tokenizer.tokenize(sentence), token_indexers)})
                          for sentence in sentences]
        self.vocab = Vocabulary.from_instances(self.instances)



    def test_lm_can_run(self):
        encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                num_characters=262,
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        text_field_embedder = BasicTextFieldEmbedder({"token_characters": encoder},
                                                     allow_unmatched_keys=True)

        lstm = torch.nn.LSTM(bidirectional=True, num_layers=3, input_size=16, hidden_size=7, batch_first=True)
        contextualizer = PytorchSeq2SeqWrapper(lstm)

        softmax = Softmax(num_words=self.vocab.get_vocab_size("tokens"),
                          embedding_dim=7)

        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)

        model = BidirectionalLanguageModel(vocab=self.vocab,
                                           text_field_embedder=text_field_embedder,
                                           contextualizer=contextualizer,
                                           softmax=softmax)

        for batch in iterator(self.instances, num_epochs=1):
            token_dict = batch['tokens']
            mask = get_text_field_mask(token_dict)
            output = text_field_embedder(token_dict)

            assert tuple(output.shape) == (2, 6, 16)

            contextualized = contextualizer(output, mask)
            assert tuple(contextualized.shape) == (2, 6, 14)

        for batch in iterator(self.instances, num_epochs=1):
            result = model(**batch)

            assert set(result) == {"loss", "forward_loss", "backward_loss"}
