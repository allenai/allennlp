from allennlp.common import cached_transformers
from allennlp.common.testing import ModelTestCase
from allennlp.models.vilbert import Nlvr2Vilbert


class TestVilbert(ModelTestCase):
    def test_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.jsonnet"
        self.ensure_model_can_train_save_and_load(param_file)

    def test_model_can_train_save_and_load_with_cache(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix=self.__class__.__name__) as d:
            overrides = {"dataset_reader": {"feature_cache_dir": str(d)}}
            import json

            overrides = json.dumps(overrides)
            param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.jsonnet"
            self.ensure_model_can_train_save_and_load(param_file, overrides=overrides)

    def test_model_loads_weights_correctly(self):
        model_name = "epwalsh/bert-xsmall-dummy"
        model = Nlvr2Vilbert.from_pretrained_module(
            pretrained_module=model_name,
            vocab=None,
            image_feature_dim=2048,
            image_num_hidden_layers=1,
            image_hidden_size=3,
            combined_hidden_size=5,
            pooled_output_dim=7,
            image_intermediate_size=11,
            image_num_attention_heads=1,
            combined_num_attention_heads=1,
            image_attention_dropout=0.0,
            image_hidden_dropout=0.0,
            v_biattention_id=[0, 1],
            t_biattention_id=[0, 1],
            fixed_t_layer=0,
            fixed_v_layer=0,
        )

        def convert_transformer_param_name(name: str):
            # We wrap the encoder in a `TimeDistributed`, which gives us this extra _module.
            name = name.replace("encoder", "encoder._module")
            name = name.replace("layer", "layers1")
            name = name.replace("LayerNorm", "layer_norm")
            return name

        transformer = cached_transformers.get(model_name, False)
        model_parameters = dict(model.named_parameters())
        transformer_parameters = dict(transformer.named_parameters())

        # We loop over the transformer parameters here, because the encoder check is easier from
        # this side (all of these encoder parameters should match, but that's not true the other way
        # around).
        for name, parameter in transformer_parameters.items():
            if name.startswith("embeddings"):
                # Embedding layer should be identical
                assert parameter.allclose(model_parameters[name])
            if name.startswith("encoder"):
                # Encoder parameters should also be identical, after we match up the names
                # correctly.
                our_name = convert_transformer_param_name(name)
                assert parameter.allclose(model_parameters[our_name])
