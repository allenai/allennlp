import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import ExecutionError
from allennlp.semparse.common import Date


class TestDate(AllenNlpTestCase):
    # pylint: disable=no-self-use
    def test_date_comparison_works(self):
        assert Date(2013, 12, 31) > Date(2013, 12, 30)
        assert Date(2013, 12, 31) == Date(2013, 12, -1)
        assert Date(2013, -1, -1) >= Date(2013, 12, 31)
        # pylint: disable=singleton-comparison
        assert (Date(2013, 12, -1) > Date(2013, 12, 31)) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert (Date(2013, 12, 31) > 2013) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert (Date(2013, 12, 31) >= 2013) == False
        with pytest.raises(ExecutionError, match='only compare Dates with Dates'):
            assert Date(2013, 12, 31) != 2013
        assert (Date(2018, 1, 1) >= Date(-1, 2, 1)) == False
        assert (Date(2018, 1, 1) < Date(-1, 2, 1)) == False
        # When year is unknown in both cases, we can compare months and days.
        assert Date(-1, 2, 1) < Date(-1, 2, 3)
        # If both year and month are not know in both cases, the comparison is undefined, and both
        # < and >= return False.
        assert (Date(-1, -1, 1) < Date(-1, -1, 3)) == False
        assert (Date(-1, -1, 1) >= Date(-1, -1, 3)) == False
        # Same when year is known, but months are not.
        assert (Date(2018, -1, 1) < Date(2018, -1, 3)) == False
        assert (Date(2018, -1, 1) >= Date(2018, -1, 3)) == False
