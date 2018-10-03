import numpy as np
import torch

from allennlp.common.testing import ModelTestCase
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
from allennlp.modules.token_embedders import TokenCharactersEncoder, Embedding
from allennlp.modules.seq2vec_encoders.cnn_highway_encoder import CnnHighwayEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training import Trainer

BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

class TestBidirectionalLM(ModelTestCase):
    def setUp(self):
        super().setUp()

        sentences = ["This is the first sentence.", "This is yet another sentence."]
        token_indexers = {
                "tokens": SingleIdTokenIndexer(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN),
                "token_characters": TokenCharactersIndexer(bos_token=BOS_TOKEN, eos_token=EOS_TOKEN)
        }
        tokenizer = WordTokenizer()

        self.instances = [Instance({"tokens": TextField(tokenizer.tokenize(sentence), token_indexers)})
                          for sentence in sentences]


        tokens_to_add = {
                "tokens": [BOS_TOKEN, EOS_TOKEN],
                "token_characters": [c for c in BOS_TOKEN + EOS_TOKEN]
        }

        self.vocab = Vocabulary.from_instances(self.instances, tokens_to_add=tokens_to_add)

    def test_lm_can_run(self):
        encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        embedding = Embedding(num_embeddings=262, embedding_dim=4)
        tce = TokenCharactersEncoder(embedding=embedding,
                                     encoder=encoder)

        text_field_embedder = BasicTextFieldEmbedder({"token_characters": tce},
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

        # Try the pieces individually
        for batch in iterator(self.instances, num_epochs=1):
            token_dict = batch['tokens']
            mask = get_text_field_mask(token_dict)
            output = text_field_embedder(token_dict)

            # Sequence length is 8 because of BOS / EOS.
            assert tuple(output.shape) == (2, 8, 16)

            contextualized = contextualizer(output, mask)
            assert tuple(contextualized.shape) == (2, 8, 14)

        # Try the whole thing
        for batch in iterator(self.instances, num_epochs=1):
            result = model(**batch)

            assert set(result) == {"loss", "forward_loss", "backward_loss", "lm_embeddings"}

            # The model should have removed the BOS / EOS tokens.
            embeddings = result["lm_embeddings"]
            assert tuple(embeddings.shape) == (2, 6, 14)

            loss = result["loss"].item()
            forward_loss = result["forward_loss"].item()
            backward_loss = result["backward_loss"].item()

            np.testing.assert_almost_equal(loss, (forward_loss + backward_loss) / 2, decimal=3)

        # Try training it
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          train_dataset=self.instances,
                          num_epochs=100)

        trainer.train()
