import torch
from torch.testing import assert_allclose
from overrides import overrides
from transformers import AutoModel
from transformers.models.albert.modeling_albert import AlbertEmbeddings

from allennlp.common import cached_transformers
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.transformer import TransformerStack, TransformerEmbeddings
from allennlp.common.testing import AllenNlpTestCase


class TestTransformerToolkit(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.vocab = Vocabulary()
        # populate vocab.
        self.vocab.add_token_to_namespace("word")
        self.vocab.add_token_to_namespace("the")
        self.vocab.add_token_to_namespace("an")

    def test_create_embedder_using_toolkit(self):

        embedding_file = str(self.FIXTURES_ROOT / "embeddings/glove.6B.300d.sample.txt.gz")

        class TinyTransformer(TokenEmbedder):
            def __init__(self, vocab, embedding_dim, hidden_size, intermediate_size):
                super().__init__()
                self.embeddings = Embedding(
                    pretrained_file=embedding_file,
                    embedding_dim=embedding_dim,
                    projection_dim=hidden_size,
                    vocab=vocab,
                )

                self.transformer = TransformerStack(
                    num_hidden_layers=4,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )

            @overrides
            def forward(self, token_ids: torch.LongTensor):
                x = self.embeddings(token_ids)
                x = self.transformer(x)
                return x

        tiny = TinyTransformer(self.vocab, embedding_dim=300, hidden_size=80, intermediate_size=40)
        tiny.forward(torch.LongTensor([[0, 1, 2]]))

    def test_use_first_four_layers_of_pretrained(self):
        pretrained = "bert-base-cased"

        class SmallTransformer(TokenEmbedder):
            def __init__(self):
                super().__init__()
                self.embeddings = TransformerEmbeddings.from_pretrained_module(
                    pretrained, relevant_module="bert.embeddings"
                )
                self.transformer = TransformerStack.from_pretrained_module(
                    pretrained,
                    num_hidden_layers=4,
                    relevant_module="bert.encoder",
                    strict=False,
                )

            @overrides
            def forward(self, token_ids: torch.LongTensor):
                x = self.embeddings(token_ids)
                x = self.transformer(x)
                return x

        small = SmallTransformer()
        assert len(small.transformer.layers) == 4
        small(torch.LongTensor([[0, 1, 2]]))

    def test_use_selected_layers_of_bert_for_different_purposes(self):
        class MediumTransformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = TransformerEmbeddings.from_pretrained_module(
                    "bert-base-cased", relevant_module="bert.embeddings"
                )
                self.separate_transformer = TransformerStack.from_pretrained_module(
                    "bert-base-cased",
                    relevant_module="bert.encoder",
                    num_hidden_layers=8,
                    strict=False,
                )
                self.combined_transformer = TransformerStack.from_pretrained_module(
                    "bert-base-cased",
                    relevant_module="bert.encoder",
                    num_hidden_layers=4,
                    mapping={f"layer.{l}": f"layers.{i}" for (i, l) in enumerate(range(8, 12))},
                    strict=False,
                )

            @overrides
            def forward(
                self,
                left_token_ids: torch.LongTensor,
                right_token_ids: torch.LongTensor,
            ):

                left = self.embeddings(left_token_ids)
                left = self.separate_transformer(left)

                right = self.embeddings(right_token_ids)
                right = self.separate_transformer(right)

                # combine the sequences in some meaningful way. here, we just add them.
                # combined = combine_masked_sequences(left, left_mask, right, right_mask)
                combined = left + right

                return self.combined_transformer(combined)

        medium = MediumTransformer()
        assert (len(medium.separate_transformer.layers)) == 8
        assert (len(medium.combined_transformer.layers)) == 4

        pretrained = cached_transformers.get("bert-base-cased", False)
        pretrained_layers = dict(pretrained.encoder.layer.named_modules())

        separate_layers = dict(medium.separate_transformer.layers.named_modules())
        assert_allclose(
            separate_layers["0"].intermediate.dense.weight.data,
            pretrained_layers["0"].intermediate.dense.weight.data,
        )

        combined_layers = dict(medium.combined_transformer.layers.named_modules())
        assert_allclose(
            combined_layers["0"].intermediate.dense.weight.data,
            pretrained_layers["8"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["1"].intermediate.dense.weight.data,
            pretrained_layers["9"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["2"].intermediate.dense.weight.data,
            pretrained_layers["10"].intermediate.dense.weight.data,
        )
        assert_allclose(
            combined_layers["3"].intermediate.dense.weight.data,
            pretrained_layers["11"].intermediate.dense.weight.data,
        )

    def test_combination_of_two_different_berts(self):
        # Regular BERT, but with AlBERT's special compressed embedding scheme

        class AlmostRegularTransformer(TokenEmbedder):
            def __init__(self):
                super().__init__()
                self.embeddings = AutoModel.from_pretrained("albert-base-v2").embeddings
                self.transformer = TransformerStack.from_pretrained_module(
                    "bert-base-cased", relevant_module="bert.encoder"
                )
                # We want to tune only the embeddings, because that's our experiment.
                self.transformer.requires_grad = False

            @overrides
            def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor):
                x = self.embeddings(token_ids, mask)
                x = self.transformer(x)
                return x

        almost = AlmostRegularTransformer()
        assert len(almost.transformer.layers) == 12
        assert isinstance(almost.embeddings, AlbertEmbeddings)
