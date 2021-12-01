"""
The transformer toolkit provides a set of reusable modules that can be used to experiment
with transformer architectures. It also simplifies the way one can take apart
the pretrained transformer weights from an existing model, and plug them in a new architecture.

Examples:

1. Create a small transformer that uses GLoVE embeddings.

```
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
```

2. Use the first 4 layers of `bert-base-uncased`.

```
pretrained = cached_transformers.get("bert-base-uncased", False)

class SmallTransformer(TokenEmbedder):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerEmbeddings.from_pretrained_module(pretrained)

        self.transformer = TransformerStack.from_pretrained_module(
            pretrained, num_hidden_layers=4
        )

    @overrides
    def forward(self, token_ids: torch.LongTensor):
        x = self.embeddings(token_ids)
        x = self.transformer(x)
        return x

small = SmallTransformer()
assert len(small.transformer.layers) == 4
small.forward(torch.LongTensor([[0, 1, 2]]))
```

3. Use the first 8 layers of `bert-base-uncased` to separately encode two text inputs, combine the representations,
and use the last 4 layers on the combined representation.

```
class MediumTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerEmbeddings.from_pretrained_module("bert-base-uncased")
        self.separate_transformer = TransformerStack.from_pretrained_module(
            "bert-base-uncased", num_hidden_layers=range(0, 8)
        )
        self.combined_transformer = TransformerStack.from_pretrained_module(
            "bert-base-uncased",
            num_hidden_layers=range(8, 12),
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

        # combine the sequences in some meaningful way.
        # Here, we just add them for simplicity. In reality,
        # concatenation may be a better option.
        combined = left + right

        return self.combined_transformer(combined)

medium = MediumTransformer()
assert (len(medium.separate_transformer.layers)) == 8
assert (len(medium.combined_transformer.layers)) == 4
```

4. Combine different flavors of BERT.

```
# Regular BERT, but with AlBERT's special compressed embedding scheme
class AlmostRegularTransformer(TokenEmbedder):
    def __init__(self):
        super().__init__()
        self.embeddings = TransformerEmbeddings.get_relevant_module("albert-base-v2")
        self.transformer = TransformerStack.from_pretrained_module("bert-base-uncased")
        # We want to tune only the embeddings, because that's our experiment.
        self.transformer.requires_grad = False

    @overrides
    def forward(self, token_ids: torch.LongTensor, mask: torch.BoolTensor):
        x = self.embeddings(token_ids, mask)
        x = self.transformer(x)
        return x

almost = AlmostRegularTransformer()
```
"""

from allennlp.modules.transformer.layer_norm import LayerNorm
from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding
from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.transformer_embeddings import (
    Embeddings,
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
)
from allennlp.modules.transformer.attention_module import SelfAttention, T5Attention
from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.transformer_layer import AttentionLayer, TransformerLayer
from allennlp.modules.transformer.transformer_stack import TransformerStack
from allennlp.modules.transformer.transformer_pooler import TransformerPooler
from allennlp.modules.transformer.output_layer import OutputLayer

from allennlp.modules.transformer.bimodal_attention import BiModalAttention
from allennlp.modules.transformer.bimodal_encoder import BiModalEncoder
from allennlp.modules.transformer.t5 import T5
