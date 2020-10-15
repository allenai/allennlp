"""
The transformer toolkit provides a set of reusable modules that can be used to experiment
with transformer architectures. It also simplifies the way one can take apart
the pretrained transformer weights from an existing model, and plug them in a new architecture.

Examples:

1. Create a small transformer that uses GLoVE embeddings.

```
from allennlp.modules.transformer import TransformerEncoder
from allennlp.modules.token_embedders.embedding import Embedding

embedding_file = os.path.join("embeddings/glove.6B.300d.sample.txt.gz")
vocab = Vocabulary()
# populate vocab.

class TinyTransformer(Model, TransformerModule):
    _huggingface_mapping = TransformerEncoder._huggingface_mapping
    def __init__(self,
                 vocab,
                 embedding_file,
                 embedding_dim: int,
                 encoder: TransformerEncoder,
                ):
        super().__init__(vocab)
        self.embedding_layer = Embedding(pretrained_file=embedding_file,
                                         embedding_dim=embedding_dim,
                                         projection_dim=encoder._hidden_size,
                                         vocab=vocab)

        self.encoder = encoder

    def forward(self, inp):
        embedded = self.embedding_layer(inp)
        outputs = self.encoder(embedded)
        return outputs

encoder = TransformerEncoder(num_hidden_layers=4,
                             hidden_size=80,
                             intermediate_size=40,
                             num_attention_heads=8,
                             attention_dropout=0.1,
                             hidden_dropout=0.1,
                             activation="relu")

tiny = TinyTransformer(vocab, embedding_file, embedding_dim=300, encoder=encoder)
```

2. Use the first 4 layers of `bert-base-uncased`.

```
from transformers.modeling_auto import AutoModel
pretrained = AutoModel.from_pretrained('bert-base-uncased')
encoder = TransformerEncoder.from_pretrained_module(pretrained.encoder, num_hidden_layers=4)

tiny = TinyTransformer(vocab, embedding_file, embedding_dim=300, encoder=encoder)
```

Alternatively, override the `from_pretrained_module` method in `TinyTransformer`.

```
    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: torch.nn.Module,
        vocab: Vocabulary,
        embedding_file: str,
        embedding_dim: int,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        encoder = TransformerEncoder.from_pretrained_module(pretrained_module.encoder, source, mapping, **kwargs)
        final_kwargs = {}
        final_kwargs["vocab"] = vocab
        final_kwargs["embedding_file"] = embedding_file
        final_kwargs["embedding_dim"] = embedding_dim
        final_kwargs["encoder"] = encoder
        return cls(**final_kwargs)

tiny = TinyTransformer.from_pretrained_module(pretrained, vocab, embedding_file, embedding_dim=300, num_hidden_layers=4)
```

3. Use the first 8 layers of `bert-base-uncased` to separately encode two text inputs, combine the representations,
and use the last 4 layers on the combined representation.

```class CombineTransformer(Model, TransformerModule):
    _huggingface_mapping = TransformerEncoder._huggingface_mapping
    def __init__(self,
                 vocab,
                 embedding_file,
                 embedding_dim: int,
                 text_encoder: TransformerEncoder,
                 combine_encoder: TransformerEncoder,
                ):
        super().__init__(vocab)
        self.embedding_layer = Embedding(pretrained_file=embedding_file,
                                         embedding_dim=embedding_dim,
                                         projection_dim=encoder._hidden_size,
                                         vocab=vocab)

        self.text_encoder = text_encoder
        self.combine_encoder = combine_encoder


    def forward(self, text1, text2):
        embedded1 = self.embedding_layer(text1)
        embedded2 = self.embedding_layer(text2)
        enc1 = self.text_encoder(embedded1)
        enc2 = self.text_encoder(embedded2)
        combined = enc1 + enc2 # Can also concat instead of add.
        outputs = self.combine_encoder(combined)
        return outputs

    @classmethod
    def from_pretrained_module(
        cls,
        pretrained_module: torch.nn.Module,
        vocab: Vocabulary,
        embedding_file: str,
        embedding_dim: int,
        source="huggingface",
        mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        num_hidden_layers = kwargs.get("num_hidden_layers", 8)
        text_encoder = TransformerEncoder.from_pretrained_module(pretrained_module.encoder,
                                                             source,
                                                             mapping,
                                                             num_hidden_layers=num_hidden_layers)

        combine_hidden_layers = kwargs.get("combine_hidden_layers", 4)
        mapping = copy.deepcopy(TransformerEncoder._huggingface_mapping)
        mapping["0"] = "8"
        mapping["1"] = "9"
        mapping["2"] = "10"
        mapping["3"] = "11"
        combine_encoder = TransformerEncoder.from_pretrained_module(pretrained_module.encoder,
                                                                    source,
                                                                    mapping,
                                                                    num_hidden_layers=combine_hidden_layers)
        final_kwargs = {}
        final_kwargs["vocab"] = vocab
        final_kwargs["embedding_file"] = embedding_file
        final_kwargs["embedding_dim"] = embedding_dim
        final_kwargs["text_encoder"] = text_encoder
        final_kwargs["combine_encoder"] = combine_encoder
        return cls(**final_kwargs)

combined_transformer = CombineTransformer.from_pretrained_module(pretrained, vocab, embedding_file, 300)
```
"""

from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding

from allennlp.modules.transformer.transformer_module import TransformerModule
from allennlp.modules.transformer.transformer_embeddings import (
    Embeddings,
    TextEmbeddings,
    TransformerEmbeddings,
    ImageFeatureEmbeddings,
)
from allennlp.modules.transformer.self_attention import SelfAttention
from allennlp.modules.transformer.activation_layer import ActivationLayer
from allennlp.modules.transformer.transformer_layer import AttentionLayer, TransformerLayer
from allennlp.modules.transformer.transformer_block import TransformerBlock
from allennlp.modules.transformer.transformer_pooler import TransformerPooler
from allennlp.modules.transformer.output_layer import OutputLayer

from allennlp.modules.transformer.bimodal_attention import BiModalAttention
from allennlp.modules.transformer.bimodal_encoder import BiModalEncoder
