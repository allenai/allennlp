"""
Database utilities for the service
"""
from typing import Optional
import json
import datetime
import logging
import os

import psycopg2

from allennlp.common.util import JsonDict
from allennlp.service.permalinks import Permadata

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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

class DemoDatabase:
    """
    This class represents a PostgreSQL database backing the demo server.
    Currently it is used to store predictions, in order to enable permalinks.
    In the future it could also be used to store user-submitted feedback about predictions.
    """
    def __init__(self, dbname: str, host: str, user: str, password: str) -> None:
        logger.info("initializing database connection:")
        logger.info("host: %s", host)
        logger.info("dbname: %s", dbname)
        self.conn = psycopg2.connect(host=host,
                                     user=user,
                                     password=password,
                                     dbname=dbname)
        logger.info("successfully initialized database connection")

    @classmethod
    def from_environment(cls) -> Optional['DemoDatabase']:
        host = os.environ.get("DEMO_POSTGRES_HOST")
        dbname = os.environ.get("DEMO_POSTGRES_DBNAME")
        user = os.environ.get("DEMO_POSTGRES_USER")
        password = os.environ.get("DEMO_POSTGRES_PASSWORD")

        if all([host, dbname, user, password]):
            logger.info("initializing demo database connection using environment variables")
            return DemoDatabase(dbname=dbname, host=host, user=user, password=password)
        else:
            logger.info("relevant environment variables not found, so no demo database")
            return None


    def add_result(self,
                   headers: JsonDict,
                   model_name: str,
                   inputs: JsonDict,
                   outputs: JsonDict) -> int:
        """
        Add the prediction to the database so that it can later
        be retrieved via permalink.
        """
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


    def get_result(self, perma_id: int) -> Optional[Permadata]:
        """
        Gets the result from the database with the given id.
        Returns ``None`` if no such result.
        """
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
