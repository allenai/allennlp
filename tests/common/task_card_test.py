from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.task_card import TaskCard


class TestTaskCard(AllenNlpTestCase):
    def test_init(self):
        task_card = TaskCard(
            id="fake_name",
            name="Fake Name",
            description="Task's description",
            expected_inputs="Passage (text string), Question (text string)",
            expected_outputs="Answer span (start token position and end token position).",
            examples=[
                {
                    "premise": "A handmade djembe was on display at the Smithsonian.",
                    "hypothesis": "Visitors could see the djembe.",
                }
            ],
        )

        assert task_card.id == "fake_name"
        assert task_card.name == "Fake Name"
        assert task_card.expected_inputs == "Passage (text string), Question (text string)"
