# pylint: disable=no-self-use,invalid-name

from allennlp.tools.drop_eval import _normalize_answer, get_metrics, evaluate_json

class TestDropEvalNormalize:
    def test_number_parse(self):
        assert _normalize_answer("12.0") == _normalize_answer("12.0  ")
        assert _normalize_answer("12.0") == _normalize_answer("12.000")
        assert _normalize_answer("12.0") == _normalize_answer("12")
        assert _normalize_answer("12.0") == _normalize_answer("  1.2e1  ")

    def test_punctations(self):
        assert _normalize_answer("12.0 persons") == "12.0 persons"
        assert _normalize_answer("S.K. Singh") == "sk singh"


class TestDropEvalGetMetrics:
    def test_float_numbers(self):
        assert get_metrics(["78"], ["78.0"]) == (1.0, 1.0)

    def test_metric_is_length_aware(self):
        # Overall F1 should be mean([1.0, 0.0])
        assert get_metrics(predicted=["td"], gold=["td", "td"]) == (0.0, 0.5)
        assert get_metrics("td", ["td", "td"]) == (0.0, 0.5)
        # Overall F1 should be mean ([1.0, 0.0]) = 0.5
        assert get_metrics(predicted=["td", "td"], gold=["td"]) == (0.0, 0.5)
        assert get_metrics(predicted=["td", "td"], gold="td") == (0.0, 0.5)

        # F1 score is mean([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        assert get_metrics(predicted=["the", "fat", "cat", "the fat", "fat cat", "the fat cat"],
                           gold=["cat"]) == (0.0, 0.17)
        assert get_metrics(predicted=["cat"],
                           gold=["the", "fat", "cat", "the fat", "fat cat", "the fat cat"]) == (0.0, 0.17)
        # F1 score is mean([1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        assert get_metrics(predicted=["the", "fat", "cat", "the fat", "fat cat", "the fat cat"],
                           gold=["cat", "cat dog"]) == (0.0, 0.25)

    def test_articles_are_ignored(self):
        assert get_metrics(["td"], ["the td"]) == (1.0, 1.0)
        assert get_metrics(["the a NOT an ARTICLE the an a"], ["NOT ARTICLE"]) == (1.0, 1.0)

    def test_f1_ignores_word_order(self):
        assert get_metrics(["John Elton"], ["Elton John"]) == (0.0, 1.0)
        assert get_metrics(["50 yard"], ["yard 50"]) == (0.0, 1.0)
        assert get_metrics(["order word right"], ["right word order"]) == (0.0, 1.0)

    def test_periods_commas_and_spaces_are_ignored(self):
        assert get_metrics(["Per.i.o.d...."], [".P....e.r,,i;;;o...d,,"]) == (1.0, 1.0)
        assert get_metrics(["Spa     c  e   s     "], ["    Spa c     e s"]) == (1.0, 1.0)

    def test_splitting_on_hyphens(self):
        assert get_metrics(["78-yard"], ["78 yard"]) == (1.0, 1.0)
        assert get_metrics(["78 yard"], ["78-yard"]) == (1.0, 1.0)
        assert get_metrics(["78"], ["78-yard"]) == (0.0, 0.67)
        assert get_metrics(["78-yard"], ["78"]) == (0.0, 0.67)

    def test_casing_is_ignored(self):
        assert get_metrics(["This was a triumph"], ["tHIS Was A TRIUMPH"]) == (1.0, 1.0)

    def test_overlap_in_correct_cases(self):
        assert get_metrics(["Green bay packers"], ["Green bay packers"]) == (1.0, 1.0)
        assert get_metrics(["Green bay", "packers"], ["Green bay", "packers"]) == (1.0, 1.0)
        assert get_metrics(["Green", "bay", "packers"], ["Green", "bay", "packers"]) == (1.0, 1.0)

    def test_simple_overlap_in_incorrect_cases(self):
        assert get_metrics([""], ["army"]) == (0.0, 0.0)
        assert get_metrics(["packers"], ["Green bay packers"]) == (0.0, 0.5)
        assert get_metrics(["packers"], ["Green bay"]) == (0.0, 0.0)
        # if the numbers in the span don't match f1 is 0
        assert get_metrics(["yard"], ["36 yard td"]) == (0.0, 0.0)
        assert get_metrics(["23 yards"], ["43 yards"]) == (0.0, 0.0)
        # however, if number matches its not given extra weight over the non-functional words
        assert get_metrics(["56 yards"], ["56 yd"]) == (0.0, 0.5)
        assert get_metrics(["26"], ["26 yard td"]) == (0.0, 0.5)

    def test_multi_span_overlap_in_incorrect_cases(self):
        # only consider bags with matching numbers if they are present
        # F1 scores of:     1.0        2/3   0.0   0.0   0.0   0.0
        # Average them to get F1 of 0.28
        assert get_metrics(["78-yard", "56", "28", "40", "44", "touchdown"],
                           ["78-yard", "56 yard", "1 yard touchdown"]) == (0.0, 0.28)

        # two copies of same value will account for only one match (using optimal 1-1 bag alignment)
        assert get_metrics(["23", "23 yard"],
                           ["23-yard", "56 yards"]) == (0.0, 0.5)

        # matching done at individual span level and not pooled into one global bag
        assert get_metrics(["John Karman", "Joe Hardy"],
                           ["Joe Karman", "John Hardy"]) == (0.0, 0.5)

        # macro-averaging F1 over spans
        assert get_metrics(["ottoman", "Kantakouzenous"],
                           ["ottoman", "army of Kantakouzenous"]) == (0.0, 0.75)

    def test_order_invariance(self):
        assert get_metrics(["a"], ["a", "b"]) == (0, 0.5)
        assert get_metrics(["b"], ["a", "b"]) == (0, 0.5)
        assert get_metrics(["b"], ["b", "a"]) == (0, 0.5)


class TestDropEvalFunctional:
    def test_json_loader(self):
        annotation = {"pid1": {"qa_pairs":[{"answer": {"number": "1"}, "validated_answers": \
                                                        [{"number": "0"}], "query_id":"qid1"}]}}
        prediction = {"qid1": "1"}
        assert evaluate_json(annotation, prediction) == (1.0, 1.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["2"]}, "validated_answers": \
                                                        [{"number": "2"}], "query_id":"qid1"}]}}
        prediction = {"qid1": "2"}
        assert evaluate_json(annotation, prediction) == (1.0, 1.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["0"]}, "validated_answers": \
                                        [{"number": "1"}, {"number": "2"}], "query_id":"qid1"}]}}
        prediction = {"qid1": "1"}
        assert evaluate_json(annotation, prediction) == (1.0, 1.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"date": {"day": "17", "month": "August", "year": ""}},\
                            "validated_answers": [{"spans": ["August"]}, {"number": "17"}], "query_id":"qid1"}]}}
        prediction = {"qid1": "17 August"}
        assert evaluate_json(annotation, prediction) == (1.0, 1.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["span1", "span2"]}, "validated_answers": \
                                        [{"spans": ["span2"]}], "query_id":"qid1"}]}}
        prediction = {"qid1": "span1"}
        assert evaluate_json(annotation, prediction) == (0.0, 0.5)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["1"]}, "validated_answers": [{"number": "0"}], \
                                            "query_id":"qid1"}]}}
        prediction = {"qid0": "2"}
        assert evaluate_json(annotation, prediction) == (0.0, 0.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["answer1"]}, "validated_answers": \
                                        [{"spans": ["answer2"]}], "query_id":"qid1"}]}}
        prediction = {"qid1": "answer"}
        assert evaluate_json(annotation, prediction) == (0.0, 0.0)

        annotation = {"pid1": {"qa_pairs":[{"answer": {"spans": ["answer1"]}, "query_id":"qid1"},\
                                        {"answer": {"spans": ["answer2"]}, "query_id":"qid2"}]}}
        prediction = {"qid1": "answer", "qid2": "answer2"}
        assert evaluate_json(annotation, prediction) == (0.5, 0.5)
