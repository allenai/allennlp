import sys

import torch
from torch.nn import Parameter

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase


def _clean_output(output: str) -> str:
    # Removes color characters.
    return (
        output.replace("\x1b[0m", "")
        .replace("\x1b[31m", "")
        .replace("\x1b[32m", "")
        .replace("\x1b[33m", "")
        .strip()
    )


class TestDiffCommand(AllenNlpTestCase):
    def test_from_archive(self, capsys):
        archive_path = str(
            self.FIXTURES_ROOT / "basic_classifier" / "serialization" / "model.tar.gz"
        )
        sys.argv = ["allennlp", "diff", archive_path, archive_path]
        main()
        captured = capsys.readouterr()
        assert (
            _clean_output(captured.out)
            == """
 _text_field_embedder.token_embedder_tokens.weight, shape = (213, 10)
 _seq2seq_encoder._module.weight_ih_l0, shape = (64, 10)
 _seq2seq_encoder._module.weight_hh_l0, shape = (64, 16)
 _seq2seq_encoder._module.bias_ih_l0, shape = (64,)
 _seq2seq_encoder._module.bias_hh_l0, shape = (64,)
 _feedforward._linear_layers.0.weight, shape = (20, 16)
 _feedforward._linear_layers.0.bias, shape = (20,)
 _classification_layer.weight, shape = (2, 20)
 _classification_layer.bias, shape = (2,)
        """.strip()
        )

    def test_from_huggingface(self, capsys):
        model_id = "hf://epwalsh/bert-xsmall-dummy/pytorch_model.bin"
        sys.argv = [
            "allennlp",
            "diff",
            model_id,
            model_id,
        ]
        main()
        captured = capsys.readouterr()
        assert (
            _clean_output(captured.out)
            == """
 embeddings.word_embeddings.weight, shape = (250, 20)
 embeddings.position_embeddings.weight, shape = (512, 20)
 embeddings.token_type_embeddings.weight, shape = (2, 20)
 embeddings.LayerNorm.weight, shape = (20,)
 embeddings.LayerNorm.bias, shape = (20,)
 encoder.layer.0.attention.self.query.weight, shape = (20, 20)
 encoder.layer.0.attention.self.query.bias, shape = (20,)
 encoder.layer.0.attention.self.key.weight, shape = (20, 20)
 encoder.layer.0.attention.self.key.bias, shape = (20,)
 encoder.layer.0.attention.self.value.weight, shape = (20, 20)
 encoder.layer.0.attention.self.value.bias, shape = (20,)
 encoder.layer.0.attention.output.dense.weight, shape = (20, 20)
 encoder.layer.0.attention.output.dense.bias, shape = (20,)
 encoder.layer.0.attention.output.LayerNorm.weight, shape = (20,)
 encoder.layer.0.attention.output.LayerNorm.bias, shape = (20,)
 encoder.layer.0.intermediate.dense.weight, shape = (40, 20)
 encoder.layer.0.intermediate.dense.bias, shape = (40,)
 encoder.layer.0.output.dense.weight, shape = (20, 40)
 encoder.layer.0.output.dense.bias, shape = (20,)
 encoder.layer.0.output.LayerNorm.weight, shape = (20,)
 encoder.layer.0.output.LayerNorm.bias, shape = (20,)
 pooler.dense.weight, shape = (20, 20)
 pooler.dense.bias, shape = (20,)
        """.strip()
        )

    def test_diff_correct(self, capsys):
        class ModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Parameter(torch.tensor([1.0, 0.0, 0.0]))
                self.b = Parameter(torch.tensor([1.0, 0.0, 0.0]))
                self.c = Parameter(torch.tensor([1.0, 0.0, 0.0]))
                self.e = Parameter(torch.tensor([1.0, 0.0, 0.0]))

        class ModelB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Parameter(torch.tensor([1.0, 0.0, 0.0]))
                self.b = Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
                self.d = Parameter(torch.tensor([1.0, 0.0, 0.0]))
                self.e = Parameter(torch.tensor([1.0, 0.0, 1.0]))

        model_a = ModelA()
        model_b = ModelB()

        torch.save(model_a.state_dict(), self.TEST_DIR / "checkpoint_a.pt")
        torch.save(model_b.state_dict(), self.TEST_DIR / "checkpoint_b.pt")
        sys.argv = [
            "allennlp",
            "diff",
            str(self.TEST_DIR / "checkpoint_a.pt"),
            str(self.TEST_DIR / "checkpoint_b.pt"),
        ]
        main()
        captured = capsys.readouterr()
        assert (
            _clean_output(captured.out)
            == """
 a, shape = (3,)
-b, shape = (3,)
-c, shape = (3,)
+b, shape = (4,)
+d, shape = (3,)
!e, shape = (3,), distance = 0.5774
        """.strip()
        )
        # NOTE: the difference value here of for 'e' of 0.5774 is currently
        # calculated at the square root of the mean squared difference between 'e'
        # in 'model_a' and 'e' in 'model_b':
        # sqrt( (0^2 + 0^2 + 1^2) / 3 ) = sqrt( 1/3 ) = 0.5774

        # Now call again with a higher theshold.
        sys.argv = [
            "allennlp",
            "diff",
            str(self.TEST_DIR / "checkpoint_a.pt"),
            str(self.TEST_DIR / "checkpoint_b.pt"),
            "--threshold",
            "0.6",
        ]
        main()
        captured = capsys.readouterr()
        assert (
            _clean_output(captured.out)
            == """
 a, shape = (3,)
-b, shape = (3,)
-c, shape = (3,)
+b, shape = (4,)
+d, shape = (3,)
 e, shape = (3,)
        """.strip()
        )

        # And call a third time with the same threshold but a higher scale.
        sys.argv = [
            "allennlp",
            "diff",
            str(self.TEST_DIR / "checkpoint_a.pt"),
            str(self.TEST_DIR / "checkpoint_b.pt"),
            "--threshold",
            "0.6",
            "--scale",
            "10.0",
        ]
        main()
        captured = capsys.readouterr()
        assert (
            _clean_output(captured.out)
            == """
 a, shape = (3,)
-b, shape = (3,)
-c, shape = (3,)
+b, shape = (4,)
+d, shape = (3,)
!e, shape = (3,), distance = 5.7735
        """.strip()
        )
