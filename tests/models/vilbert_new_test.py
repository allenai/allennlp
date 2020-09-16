from transformers.modeling_auto import AutoModel

from allennlp.common.testing import ModelTestCase
from allennlp.models.vilbert_new import Nlvr2Vilbert


class TestVilbert(ModelTestCase):
    def test_model_can_train_save_and_load(self):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment.json"
        self.ensure_model_can_train_save_and_load(param_file)

    def test_model_can_train_save_and_load_from_huggingface(self):
        param_file = self.FIXTURES_ROOT / "vilbert" / "experiment_from_huggingface.jsonnet"
        self.ensure_model_can_train_save_and_load(param_file)

    def test_model_loads_weights_correctly(self):
        model_name = "epwalsh/bert-xsmall-dummy"
        transformer = AutoModel.from_pretrained(model_name)
        model = Nlvr2Vilbert.from_pretrained_module(
            transformer,
            vocab=None,
            image_feature_dim=2048,
            image_num_hidden_layers=1,
            image_hidden_size=3,
            combined_hidden_size=5,
            pooled_output_dim=7,
            image_intermediate_size=11,
            image_attention_dropout=0.0,
            image_hidden_dropout=0.0,
            v_biattention_id=[0, 1],
            t_biattention_id=[0, 1],
            fixed_t_layer=0,
            fixed_v_layer=0,
        )

        # model = Nlvr2Vilbert.from_huggingface_model_name(
        #     vocab=None,
        #     model_name=model_name,
        #     image_feature_dim=2048,
        #     image_num_hidden_layers=1,
        #     image_hidden_size=3,
        #     combined_hidden_size=5,
        #     pooled_output_dim=7,
        #     image_intermediate_size=11,
        #     image_attention_dropout=0.0,
        #     image_hidden_dropout=0.0,
        #     v_biattention_id=[0, 1],
        #     t_biattention_id=[0, 1],
        #     fixed_t_layer=0,
        #     fixed_v_layer=0,
        # )

        def convert_transformer_param_name(name: str):
            # We wrap the encoder in a `TimeDistributed`, which gives us this extra _module.
            name = name.replace("encoder", "encoder._module")
            name = name.replace("LayerNorm", "layer_norm")
            name = name.replace(".layer.", ".layers1.")
            return name

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
