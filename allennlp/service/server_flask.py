"""
The ``server_flask`` application launches a server
that exposes trained models via a REST API,
and that includes a web interface for exploring
their predictions.

You can run this on the command line with

.. code-block:: bash

    $ python -m allennlp.service.server_flask -h
    usage: server_flask.py [-h] [--port PORT]

    Run the web service, which provides an HTTP API as well as a web demo.

    optional arguments:
      -h, --help   show this help message and exit
      --port PORT  the port to run the server on
"""
import argparse
from datetime import datetime
from typing import Dict, Optional
import json
import logging
import os
import sys
import time
from functools import lru_cache

from flask import Flask, request, Response, jsonify, send_file, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

import psycopg2

import pytz

from allennlp.common.util import JsonDict, peak_memory_mb
from allennlp.models.archival import load_archive
from allennlp.service.db import DemoDatabase, PostgresDemoDatabase
from allennlp.service.permalinks import int_to_slug, slug_to_int
from allennlp.predictors import Predictor

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("FLASK_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        return error_dict

class DemoModel:
    """
    A demo model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor
    """
    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)

def run(port: int,
        trained_models: Dict[str, DemoModel],
        static_dir: str = None) -> None:
    """Run the server programatically"""
    logger.info("Starting a flask server on port %i.", port)

    if port != 8000:
        logger.warning("The demo requires the API to be run on port 8000.")

    # This will be ``None`` if all the relevant environment variables are not defined or if
    # there is an exception when connecting to the database.
    demo_db = PostgresDemoDatabase.from_environment()

    app = make_app(static_dir, demo_db)
    CORS(app)

    for name, demo_model in trained_models.items():
        predictor = demo_model.predictor()
        app.predictors[name] = predictor

    http_server = WSGIServer(('0.0.0.0', port), app)
    logger.info("Server started on port %i.  Please visit: http://localhost:%i", port, port)
    http_server.serve_forever()

def make_app(build_dir: str = None, demo_db: Optional[DemoDatabase] = None) -> Flask:
    if build_dir is None:
        # Need path to static assets to be relative to this file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        build_dir = os.path.join(dir_path, '../../demo/build')

    if not os.path.exists(build_dir):
        logger.error("app directory %s does not exist, aborting", build_dir)
        sys.exit(-1)

    app = Flask(__name__)  # pylint: disable=invalid-name
    start_time = datetime.now(pytz.utc)
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    app.predictors = {}

    try:
        cache_size = int(CACHE_SIZE)  # type: ignore
    except ValueError:
        logger.warning("unable to parse cache size %s as int, disabling cache", CACHE_SIZE)
        cache_size = 0

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @lru_cache(maxsize=cache_size)
    def _caching_prediction(model: Predictor, data: str) -> JsonDict:
        """
        Just a wrapper around ``model.predict_json`` that allows us to use a cache decorator.
        """
        return model.predict_json(json.loads(data))

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        return send_file(os.path.join(build_dir, 'index.html'))

    @app.route('/permadata', methods=['POST', 'OPTIONS'])
    def permadata() -> Response:  # pylint: disable=unused-variable
        """
        If the user requests a permalink, the front end will POST here with the payload
            { slug: slug }
        which we convert to an integer id and use to retrieve saved results from the database.
        """
        # This is just CORS boilerplate.
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        # If we don't have a database configured, there are no permalinks.
        if demo_db is None:
            raise ServerError('Permalinks are not enabled', 400)

        # Convert the provided slug to an integer id.
        slug = request.get_json()["slug"]
        perma_id = slug_to_int(slug)
        if perma_id is None:
            # Malformed slug
            raise ServerError("Unrecognized permalink: {}".format(slug), 400)

        # Fetch the results from the database.
        try:
            permadata = demo_db.get_result(perma_id)
        except psycopg2.Error:
            logger.exception("Unable to get results from database: perma_id %s", perma_id)
            raise ServerError('Database trouble', 500)

        if permadata is None:
            # No data found, invalid id?
            raise ServerError("Unrecognized permalink: {}".format(slug), 400)

        return jsonify({
                "modelName": permadata.model_name,
                "requestData": permadata.request_data,
                "responseData": permadata.response_data
        })

    @app.route('/predict/<model_name>', methods=['POST', 'OPTIONS'])
    def predict(model_name: str) -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        # Do log if no argument is specified
        record_to_database = request.args.get("record", "true").lower() != "false"

        # Do use the cache if no argument is specified
        use_cache = request.args.get("cache", "true").lower() != "false"

        model = app.predictors.get(model_name.lower())
        if model is None:
            raise ServerError("unknown model: {}".format(model_name), status_code=400)

        data = request.get_json()

        log_blob = {"model": model_name, "inputs": data, "cached": False, "outputs": {}}

        # Record the number of cache hits before we hit the cache so we can tell whether we hit or not.
        # In theory this could result in false positives.
        pre_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        if use_cache and cache_size > 0:
            # lru_cache insists that all function arguments be hashable,
            # so unfortunately we have to stringify the data.
            prediction = _caching_prediction(model, json.dumps(data))
        else:
            # if cache_size is 0, skip caching altogether
            prediction = model.predict_json(data)

        post_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        if record_to_database and demo_db is not None:
            try:
                perma_id = None
                perma_id = demo_db.add_result(headers=dict(request.headers),
                                              model_name=model_name,
                                              inputs=data,
                                              outputs=prediction)
                if perma_id is not None:
                    slug = int_to_slug(perma_id)
                    prediction["slug"] = slug
                    log_blob["slug"] = slug

            except Exception:  # pylint: disable=broad-except
                # TODO(joelgrus): catch more specific errors
                logger.exception("Unable to add result to database", exc_info=True)

        if use_cache and post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            log_blob["cached"] = True
            time.sleep(0.25)

        # The model predictions are extremely verbose, so we only log the most human-readable
        # parts of them.
        if model_name == "machine-comprehension":
            log_blob["outputs"]["best_span_str"] = prediction["best_span_str"]
        elif model_name == "coreference-resolution":
            log_blob["outputs"]["clusters"] = prediction["clusters"]
            log_blob["outputs"]["document"] = prediction["document"]
        elif model_name == "textual-entailment":
            log_blob["outputs"]["label_probs"] = prediction["label_probs"]
        elif model_name == "named-entity-recognition":
            log_blob["outputs"]["tags"] = prediction["tags"]
        elif model_name == "semantic-role-labeling":
            verbs = []
            for verb in prediction["verbs"]:
                # Don't want to log boring verbs with no semantic parses.
                good_tags = [tag for tag in verb["tags"] if tag != "0"]
                if len(good_tags) > 1:
                    verbs.append({"verb": verb["verb"], "description": verb["description"]})
            log_blob["outputs"]["verbs"] = verbs

        elif model_name == "constituency-parsing":
            log_blob["outputs"]["trees"] = prediction["trees"]

        logger.info("prediction: %s", json.dumps(log_blob))

        print(log_blob)

        return jsonify(prediction)

    @app.route('/models')
    def list_models() -> Response:  # pylint: disable=unused-variable
        """list the available models"""
        return jsonify({"models": list(app.predictors.keys())})

    @app.route('/info')
    def info() -> Response:  # pylint: disable=unused-variable
        """List metadata about the running webserver"""
        uptime = str(datetime.now(pytz.utc) - start_time)
        git_version = os.environ.get('SOURCE_COMMIT') or ""
        return jsonify({
                "start_time": start_time_str,
                "uptime": uptime,
                "git_version": git_version,
                "peak_memory_mb": peak_memory_mb(),
                "githubUrl": "http://github.com/allenai/allennlp/commit/" + git_version})

    # As a SPA, we need to return index.html for /model-name and /model-name/permalink
    @app.route('/semantic-role-labeling')
    @app.route('/constituency-parsing')
    @app.route('/machine-comprehension')
    @app.route('/textual-entailment')
    @app.route('/coreference-resolution')
    @app.route('/named-entity-recognition')
    @app.route('/semantic-role-labeling/<permalink>')
    @app.route('/constituency-parsing/<permalink>')
    @app.route('/machine-comprehension/<permalink>')
    @app.route('/textual-entailment/<permalink>')
    @app.route('/coreference-resolution/<permalink>')
    @app.route('/named-entity-recognition/<permalink>')
    def return_page(permalink: str = None) -> Response:  # pylint: disable=unused-argument, unused-variable
        """return the page"""
        return send_file(os.path.join(build_dir, 'index.html'))

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response: # pylint: disable=unused-variable
        return send_from_directory(build_dir, path)

    @app.route('/static/js/<path:path>')
    def static_js_proxy(path: str) -> Response: # pylint: disable=unused-variable
        return send_from_directory(os.path.join(build_dir, 'static/js'), path)

    return app

# This maps from the name of the task
# to the ``DemoModel`` indicating the location of the trained model
# and the type of the ``Predictor``.  This is necessary, as you might
# have multiple models (for example, a NER tagger and a POS tagger)
# that have the same ``Predictor`` wrapper. The corresponding model
# will be served at the `/predict/<name-of-task>` API endpoint.
DEFAULT_MODELS = {
        'machine-comprehension': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz',  # pylint: disable=line-too-long
                'machine-comprehension'
        ),
        'semantic-role-labeling': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz', # pylint: disable=line-too-long
                'semantic-role-labeling'
        ),
        'textual-entailment': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz',  # pylint: disable=line-too-long
                'textual-entailment'
        ),
        'coreference-resolution': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',  # pylint: disable=line-too-long
                'coreference-resolution'
        ),
        'named-entity-recognition': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz',  # pylint: disable=line-too-long
                'sentence-tagger'
        ),
        'constituency-parsing': DemoModel(
                'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',  # pylint: disable=line-too-long
                'constituency-parser'
        )
}

def main(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)

    # pylint: disable=protected-access
    description = '''Run the web service, which provides an HTTP API as well as a web demo.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--port', type=int, default=8000, help="the port to run the server on")

    args = parser.parse_args()
    run(args.port, DEFAULT_MODELS)

if __name__ == "__main__":
    main(sys.argv[1:])
