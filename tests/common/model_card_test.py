from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.model_card import ModelCard, ModelUsage, IntendedUse, Paper
from allennlp.models import Model


class TestPretrainedModelConfiguration(AllenNlpTestCase):
    def test_init(self):
        model_card = ModelCard(
            id="fake_name",
            display_name="Fake Name",
            model_details="Model's description",
            model_usage=ModelUsage(**{"archive_file": "fake.tar.gz", "overrides": {}}),
        )

        assert model_card.id == "fake_name"
        assert model_card.display_name == "Fake Name"
        assert model_card.model_usage.archive_file == ModelUsage._storage_location + "fake.tar.gz"
        assert model_card.model_details.description == "Model's description"

    def test_init_registered_model(self):
        @Model.register("fake-model")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        model_card = ModelCard(**{"id": "this-fake-model", "registered_model_name": "fake-model"})

        assert model_card.display_name == "FakeModel"
        assert model_card.model_details.description == "This is a fake model with a docstring."

    def test_init_dict_model(self):
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        model_card = ModelCard(**{"id": "this-fake-model", "model_class": FakeModel})

        assert model_card.display_name == "FakeModel"
        assert model_card.model_details.description == "This is a fake model with a docstring."

    def test_init_registered_model_override(self):
        @Model.register("fake-model-2")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        model_card = ModelCard(
            **{
                "id": "this-fake-model",
                "registered_model_name": "fake-model-2",
                "model_details": "This is the fake model trained on a dataset.",
                "model_class": FakeModel,
            }
        )

        assert (
            model_card.model_details.description == "This is the fake model trained on a dataset."
        )

    def test_init_model_card_info_obj(self):
        @Model.register("fake-model-3")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        intended_use = IntendedUse("Use 1", "User 1")

        model_card = ModelCard(
            **{
                "id": "this-fake-model",
                "registered_model_name": "fake-model-3",
                "intended_use": intended_use,
            }
        )

        model_card_dict = model_card.to_dict()
        assert model_card.display_name == "FakeModel"

        for key, val in intended_use.__dict__.items():
            if val:
                assert key in model_card_dict
            else:
                assert key not in model_card_dict

    def test_nested_json(self):
        @Model.register("fake-model-4")
        class FakeModel(Model):
            """
            This is a fake model with a docstring.

            # Parameters

            fake_param1: str
            fake_param2: int
            """

            def forward(self, **kwargs):
                return {}

        model_card = ModelCard.from_params(
            Params(
                {
                    "id": "this-fake-model",
                    "registered_model_name": "fake-model-4",
                    "model_details": {
                        "description": "This is the fake model trained on a dataset.",
                        "paper": {
                            "title": "paper name",
                            "url": "paper link",
                            "citation": "test citation",
                        },
                    },
                    "training_data": {"dataset": {"name": "dataset 1", "url": "dataset url"}},
                }
            )
        )

        assert isinstance(model_card.model_details.paper, Paper)
        assert model_card.model_details.paper.url == "paper link"
        assert model_card.training_data.dataset.name == "dataset 1"
