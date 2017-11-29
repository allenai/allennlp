"""
Database utilities for the service
"""
from typing import Optional, List
import json
import datetime
import logging
import os

import psycopg2

from allennlp.common.util import JsonDict
from allennlp.service.permalinks import Permadata

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DemoDatabase:
    """
    This class represents a database backing the demo server.
    Currently it is used to store predictions, in order to enable permalinks.
    In the future it could also be used to store user-submitted feedback about predictions.
    """
    def add_result(self,
                   headers: JsonDict,
                   model_name: str,
                   inputs: JsonDict,
                   outputs: JsonDict) -> Optional[int]:
        """
        Add the prediction to the database so that it can later
        be retrieved via permalink.
        """
        raise NotImplementedError

    def get_result(self, perma_id: int) -> Permadata:
        """
        Gets the result from the database with the given id.
        Returns ``None`` if no such result.
        """
        raise NotImplementedError

    @classmethod
    def from_environment(cls) -> Optional['DemoDatabase']:
        """
        Instantiate a database using parameters (host, port, user, password, etc...) from environment variables.
        """
        raise NotImplementedError


# SQL for inserting predictions into the database.
INSERT_SQL = (
        """
        INSERT INTO queries (model_name, headers, request_data, response_data, timestamp)
        VALUES (%(model_name)s, %(headers)s, %(request_data)s, %(response_data)s, %(timestamp)s)
        RETURNING id
        """
)

# SQL for retrieving a prediction from the database.
RETRIEVE_SQL = (
        """
        SELECT model_name, request_data, response_data
        FROM queries
        WHERE id = (%s)
        """
)

class PostgresDemoDatabase(DemoDatabase):
    """
    Concrete Postgres implementation.
    """
    def __init__(self, dbname: str, host: str, port: str, user: str, password: str) -> None:
        self.dbname = dbname
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.conn: Optional[psycopg2.extensions.connection] = None
        self._connect()

    def _connect(self) -> None:
        logger.info("initializing database connection:")
        logger.info("host: %s", self.host)
        logger.info("port: %s", self.port)
        logger.info("dbname: %s", self.dbname)
        try:
            self.conn = psycopg2.connect(host=self.host,
                                         port=self.port,
                                         user=self.user,
                                         password=self.password,
                                         dbname=self.dbname,
                                         connect_timeout=5)
            self.conn.set_session(autocommit=True)
            logger.info("successfully initialized database connection")
        except psycopg2.Error as error:
            logger.exception("unable to connect to database")
            raise error

    def _health_check(self) -> None:
        """
        Postgres has no way of automatically reconnecting lost database
        connections. Because our database load is pretty low, we can afford to do
        a health check before each database request and (try to) reconnect if the
        connection has been lost.
        """
        try:
            with self.conn.cursor() as curs:
                # Run a simple query
                curs.execute("""SELECT 1""")
                curs.fetchone()
        except (psycopg2.Error, AttributeError):
            logger.exception("Database connection lost, reconnecting")
            self._connect()

    @classmethod
    def from_environment(cls) -> Optional['PostgresDemoDatabase']:
        host = os.environ.get("DEMO_POSTGRES_HOST")
        port = os.environ.get("DEMO_POSTGRES_PORT") or "5432"
        dbname = os.environ.get("DEMO_POSTGRES_DBNAME")
        user = os.environ.get("DEMO_POSTGRES_USER")
        password = os.environ.get("DEMO_POSTGRES_PASSWORD")

        if all([host, port, dbname, user, password]):
            try:
                logger.info("Initializing demo database connection using environment variables")
                return PostgresDemoDatabase(dbname=dbname, host=host, port=port, user=user, password=password)
            except psycopg2.Error:
                logger.exception("unable to connect to database, permalinks not enabled")
                return None
        else:
            logger.info("Relevant environment variables not found, so no demo database")
            return None


    def add_result(self,
                   headers: JsonDict,
                   model_name: str,
                   inputs: JsonDict,
                   outputs: JsonDict) -> Optional[int]:
        try:
            self._health_check()
            with self.conn.cursor() as curs:
                logger.info("inserting into the database")

                curs.execute(INSERT_SQL,
                             {'model_name'   : model_name,
                              'headers'      : json.dumps(headers),
                              'request_data' : json.dumps(inputs),
                              'response_data': json.dumps(outputs),
                              'timestamp'    : datetime.datetime.now()})

                perma_id = curs.fetchone()[0]
                logger.info("received perma_id %s", perma_id)

            return perma_id
        except (psycopg2.Error, AttributeError):
            logger.exception("Unable to insert permadata")
            return None

    def get_result(self, perma_id: int) -> Optional[Permadata]:
        try:
            self._health_check()
            with self.conn.cursor() as curs:
                logger.info("retrieving perma_id %s from database", perma_id)
                curs.execute(RETRIEVE_SQL, (perma_id,))
                row = curs.fetchone()

            # If there's no result, return None.
            if row is None:
                return None

            # Otherwise, return a ``Permadata`` instance.
            model_name, request_data, response_data = row
            return Permadata(model_name, json.loads(request_data), json.loads(response_data))
        except (psycopg2.Error, AttributeError):
            logger.exception("Unable to retrieve result")
            return None

class InMemoryDemoDatabase(DemoDatabase):
    """
    This is just for unit tests, please don't use it in production.
    """
    def __init__(self):
        self.data: List[Permadata] = []

    def add_result(self,
                   headers: JsonDict,
                   model_name: str,
                   inputs: JsonDict,
                   outputs: JsonDict) -> Optional[int]:
        self.data.append(Permadata(model_name, inputs, outputs))
        return len(self.data) - 1

    def get_result(self, perma_id: int) -> Permadata:
        try:
            return self.data[perma_id]
        except IndexError:
            return None

    @classmethod
    def from_environment(cls) -> Optional['InMemoryDemoDatabase']:
        return InMemoryDemoDatabase()
