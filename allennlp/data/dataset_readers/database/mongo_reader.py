from typing import Iterable

from pymongo import MongoClient

from allennlp.data.dataset_readers.database.database_reader import DatabaseReader


class MongoDatabaseReader(DatabaseReader):
    """
    """

    def __init__(self, database, collection, lazy: bool) -> None:
        """

        :param database:
        :param collection:
        :param lazy:
        """
        super().__init__(lazy=lazy)

        self.database = database
        self.collection = collection

    def _read(self, url: str, query: object, *args) -> Iterable:
        """

        :param url:
        :param query:
        :param args:
        :return:
        """
        client = MongoClient(url)
        results = client[self.database][self.collection].find(query)
        for result in results:
            yield self.text_to_instance(result)
