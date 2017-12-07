import os
import shelve

from allennlp.common.archival import Archivable
from allennlp.common.testing import AllenNlpTestCase

class A(Archivable):
    def __init__(self, b_name: str):
        self.b = B(b_name)
        self.name = "a-allennlp"
        self.score = 0

    def stuff_to_archive(self):
        return {"name": self.name, "score": self.score}

    def populate_stuff_from_archive(self, stuff):
        self.name = stuff["name"]
        self.score = stuff["score"]


class B(Archivable):
    def __init__(self, name: str = None):
        self.name = name

    def stuff_to_archive(self):
        return {"name": self.name}

    def populate_stuff_from_archive(self, stuff):
        self.name = stuff["name"]


class TestArchival(AllenNlpTestCase):
    def test_archival(self):
        shelf_file = os.path.join(self.TEST_DIR, "shelf")

        a = A(b_name = "b-allennlp")
        a.score = 100

        collection = a.collect(prefix='a')
        assert collection == {
                "a": {"name": "a-allennlp", "score": 100},
                "a.b": {"name": "b-allennlp"}
        }

        with shelve.open(shelf_file) as db:
            for k, v in collection.items():
                db[k] = v

        a2 = A(b_name = "junk")

        assert a2.score == 0
        assert a2.b.name == "junk"

        with shelve.open(shelf_file) as db:
            collection = dict(db)

        a2.populate_from_collection(collection, prefix='a')

        assert a2.score == 100
        assert a2.b.name == "b-allennlp"

