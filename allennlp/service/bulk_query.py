"""
this is just a standalone tool for testing concurrency of served models
"""

import json
from multiprocessing import Pool
from typing import Tuple
import datetime

import requests

HOST = "http://localhost"
PORT = "5001"
ENDPOINT = "predict"
MODEL_NAME = "pytorch"

url = "{}:{}/{}/{}".format(HOST, PORT, ENDPOINT, MODEL_NAME)  # pylint: disable=invalid-name

NUM_REQUESTS = 1000
NUM_PROCESSES = 16

def post(i: int) -> Tuple[int, float]:  # pylint: disable=redefined-outer-name
    payload = {'input': i}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    output = response.json()["output"]
    total = sum(v for row in output for v in row)
    return (i, total)

if __name__ == "__main__":
    pool = Pool(processes=NUM_PROCESSES)  # pylint: disable=invalid-name

    start = datetime.datetime.now()  # pylint: disable=invalid-name

    for result in pool.imap_unordered(post, range(NUM_REQUESTS)):
        print(result)

    end = datetime.datetime.now()  # pylint: disable=invalid-name

    print("elapsed", end - start)
