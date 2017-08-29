# pylint: disable=no-self-use,invalid-name
from typing import List
from os.path import join

from overrides import overrides

from allennlp.data.dataset_readers import SquadSentenceSelectionReader
from allennlp.data.fields import IndexField, ListField
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer


class TestSquadSentenceSelectionReader(AllenNlpTestCase):
    @overrides
    def setUp(self):
        super(TestSquadSentenceSelectionReader, self).setUp()
        # write a SQuAD json file.
        # pylint: disable=bad-continuation
        self.sentences = [
                "Architecturally, the school has a Catholic character.",
                "Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
                "Immediately in front of the Main Building and facing it, is a copper statue of "
                    "Christ with arms upraised with the legend \\\"Venite Ad Me Omnes\\\".",
                "Next to the Main Building is the Basilica of the Sacred Heart.",
                "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
                "It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly "
                    "appeared to Saint Bernadette Soubirous in 1858.",
                "At the end of the main drive (and in a direct line that connects through 3 "
                    "statues and the Gold Dome), is a simple, modern stone statue of Mary.",
                "This is another sentence.",
                "And another one.",
                "Yet another sentence 1.",
                "Yet another sentence 2.",
                "Yet another sentence 3.",
                "Yet another sentence 4.",
                "Yet another sentence 5.",
                ]
        # pylint: enable=bad-continuation
        self.passage1 = " ".join(self.sentences[:7])
        self.passage2 = " ".join(self.sentences[7:])
        self.question0 = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
        self.question1 = "What is in front of the Notre Dame Main Building?"
        self.questions = [self.question0, self.question1]
        json_string = """
        {
          "data":[
            {
              "title":"University_of_Notre_Dame",
              "paragraphs":[
                {
                  "context":"%s",
                  "qas":[
                    {
                      "answers":[
                        {
                          "answer_start":515,
                          "text":"Saint Bernadette Soubirous"
                        }
                      ],
                      "question":"%s",
                      "id":"5733be284776f41900661182"
                    },
                    {
                      "answers":[
                        {
                          "answer_start":188,
                          "text":"a copper statue of Christ"
                        }
                      ],
                      "question":"%s",
                      "id":"5733be284776f4190066117f"
                    }
                  ]
                },
                {
                  "context":"%s",
                  "qas":[ ]
                }
              ]
            }
          ]
        }
        """ % (self.passage1, self.question0, self.question1, self.passage2)
        self.tokenizer = WordTokenizer()
        self.squad_file = join(self.TEST_DIR, "squad_data.json")
        with open(self.squad_file, "w") as f:
            f.write(json_string)

    def assert_list_field_contains_correct_sentences(self, list_field: ListField, sentences: List[str]):
        expected_tokens = set()
        for sentence in sentences:
            sentence_tokens = tuple(self.tokenizer.tokenize(sentence.replace("\\\"", "\""))[0])
            expected_tokens.add(sentence_tokens)
        actual_tokens = set()
        for field in list_field.field_list:
            actual_tokens.add(tuple(field.tokens))
        assert expected_tokens == actual_tokens

    def assert_index_field_points_to_correct_sentence(self, index_field: IndexField, sentence: str):
        sentence_tokens = tuple(self.tokenizer.tokenize(sentence.replace("\\\"", "\""))[0])
        tokens_list = [tuple(field.tokens) for field in index_field.sequence_field.field_list]
        assert index_field.sequence_index == tokens_list.index(sentence_tokens)

    def test_default_squad_sentence_selection_reader(self):
        reader = SquadSentenceSelectionReader()
        instances = reader.read(self.squad_file).instances
        assert instances[0].fields["question"].tokens == self.tokenizer.tokenize(self.question0)[0]
        self.assert_list_field_contains_correct_sentences(instances[0].fields["sentences"],
                                                          self.sentences[:7])
        self.assert_index_field_points_to_correct_sentence(instances[0].fields['correct_sentence'],
                                                           self.sentences[5])
        assert instances[1].fields["question"].tokens == self.tokenizer.tokenize(self.question1)[0]
        self.assert_list_field_contains_correct_sentences(instances[1].fields["sentences"],
                                                          self.sentences[:7])
        self.assert_index_field_points_to_correct_sentence(instances[1].fields['correct_sentence'],
                                                           self.sentences[2])

    def test_negative_question_choice_works(self):
        reader = SquadSentenceSelectionReader(negative_sentence_selection="question")
        instances = reader.read(self.squad_file).instances
        self.assert_list_field_contains_correct_sentences(instances[0].fields["sentences"],
                                                          [self.sentences[5], self.question0])
        self.assert_index_field_points_to_correct_sentence(instances[0].fields['correct_sentence'],
                                                           self.sentences[5])
        self.assert_list_field_contains_correct_sentences(instances[1].fields["sentences"],
                                                          [self.sentences[2], self.question1])
        self.assert_index_field_points_to_correct_sentence(instances[1].fields['correct_sentence'],
                                                           self.sentences[2])

    def test_negative_random_question_choice_works(self):
        reader = SquadSentenceSelectionReader(negative_sentence_selection="questions-random-2")
        instances = reader.read(self.squad_file).instances
        self.assert_list_field_contains_correct_sentences(instances[0].fields["sentences"],
                                                          [self.sentences[5], self.question0, self.question1])
        self.assert_index_field_points_to_correct_sentence(instances[0].fields['correct_sentence'],
                                                           self.sentences[5])
        self.assert_list_field_contains_correct_sentences(instances[1].fields["sentences"],
                                                          [self.sentences[2], self.question0, self.question1])
        self.assert_index_field_points_to_correct_sentence(instances[1].fields['correct_sentence'],
                                                           self.sentences[2])

    def test_negative_random_and_pad_work(self):
        # We aren't going to try to guess _which_ random sentences get selected, but we will at
        # least make sure that we get the expected number of results.
        reader = SquadSentenceSelectionReader(negative_sentence_selection="random-2,pad-to-5")
        instances = reader.read(self.squad_file).instances
        assert instances[0].fields['sentences'].sequence_length() == 6
        self.assert_index_field_points_to_correct_sentence(instances[0].fields['correct_sentence'],
                                                           self.sentences[5])
        assert instances[1].fields['sentences'].sequence_length() == 6
        self.assert_index_field_points_to_correct_sentence(instances[1].fields['correct_sentence'],
                                                           self.sentences[2])
