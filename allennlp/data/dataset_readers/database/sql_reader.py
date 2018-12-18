from typing import Iterable

from sqlalchemy import create_engine

from allennlp.data.dataset_readers.database.database_reader import DatabaseReader


class PostgresDatabaseReader(DatabaseReader):
    """
    """

    def __init__(self, lazy) -> None:
        """

        :param url:
        :param query:
        :param lazy:
        :param texts:
        :param target:
        :param args:
        """
        super().__init__(lazy=lazy)

    def _read(self, url: str, query: str, *args) -> Iterable:
        """

        :return:
        """

        engine = create_engine(url)
        with engine.connect() as connection:
            self.results = connection.execute(query, args)

        for result in self.results:
            yield self.text_to_instance(dict(zip(result.keys(), result)))
