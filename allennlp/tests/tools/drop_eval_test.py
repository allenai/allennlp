from allennlp.tools.drop_eval import normalize_answer, get_metrics


class TestNormalize:
    def test_number_parse(self):
        assert normalize_answer("12.0") == normalize_answer("12.0  ")
        assert normalize_answer("12.0") == normalize_answer("12.000")
        assert normalize_answer("12.0") == normalize_answer("12")
        assert normalize_answer("12.0") == normalize_answer("  1.2e1  ")

    def test_punctations(self):
        assert normalize_answer("12.0 persons") == "12.0 persons"
        assert normalize_answer("S.K. Singh") == "sk singh"


class TestGetMetrics:
    def test_articles_are_ignored(self):
        assert get_metrics(["td"], ["the td"]) == (1.0, 1.0)
        assert get_metrics(["the a NOT an ARTICLE the an a"], ["NOT ARTICLE"]) == (1.0, 1.0)

    def test_f1_respects_word_order(self):
        # I'm not sure if this is a good idea or not; just adding it for discussion.
        assert get_metrics(["John Elton"], ["Elton John"]) == (0.0, 0.0)
        assert get_metrics(["50 yard"], ["yard 50"]) == (0.0, 0.0)
        assert get_metrics(["order word right"], ["right word order"]) == (0.0, 0.0)

    def test_periods_commas_and_spaces_are_ignored(self):
        assert get_metrics(["Per.i.o.d...."], [".P....e.r,,i;;;o...d,,"]) == (1.0, 1.0)
        assert get_metrics(["Spa     c  e   s     "], ["    Spa c     e s"]) == (1.0, 1.0)

    def test_splitting_on_hyphens(self):
        assert get_metrics(["78-yard"], ["78 yard"]) == (1.0, 1.0)
        assert get_metrics(["78 yard"], ["78-yard"]) == (1.0, 1.0)
        assert get_metrics(["78"], ["78-yard"]) == (0.0, 0.5)
        assert get_metrics(["78-yard"], ["78"]) == (0.0, 0.5)

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
        # I'd like these to be lower, but not sure how to accomplish that.  Maybe if the answer has
        # a number, you weight those tokens more heavily?
        assert get_metrics(["yard"], ["36 yard td"]) == (0.0, 0.5)
        assert get_metrics(["23 yards"], ["43 yards"]) == (0.0, 0.5)
        # Similarly, I'd like these to be higher, which could be accomplished the same way as
        # above.
        assert get_metrics(["56 yards"], ["56 yd"]) == (0.0, 0.5)
        assert get_metrics(["26"], ["26 yard td"]) == (0.0, 0.5)

    def test_multi_span_overlap_in_incorrect_cases(self):
        # I'm not sure what the values here should be, just adding some interesting cases.
        assert get_metrics(["78-yard", "56", "28", "40", "44", "touchdown"],
                           ["78-yard", "56 yard", "1 yard touchdown"]) == (0.0, 0.5)
        assert get_metrics(["ottoman", "Kantakouzenous"],
                           ["ottoman", "army of Kantakouzenous"]) == (0.0, 0.5)

        # EM obviously 0 here; not clear what F1 should be, and the decision depends on the word
        # order test above.  If we ignore word order, F1 could reasonably be 1.0 here.
        assert get_metrics(["span one", "span two"],
                           ["two span", "one span"]) == (0.0, 0.0)

        # F1 should definitely not be 1.0 in this case.
        assert get_metrics(["John Karman", "Joe Hardy"],
                           ["Joe Karman", "John Hardy"]) == (0.0, 0.0)
