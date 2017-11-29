"""
A `Sanic <http://sanic.readthedocs.io/en/latest/>`_ server that serves up
AllenNLP models as well as our demo.

Usually you would use :mod:`~allennlp.commands.serve`
rather than instantiating an ``app`` yourself.
"""
from datetime import datetime
from typing import Dict, Optional
import asyncio
import json
import logging
import os
import sys
from functools import lru_cache

from sanic import Sanic, response, request
from sanic.exceptions import ServerError
from sanic_cors import CORS

import psycopg2

import pytz

from allennlp.common.util import JsonDict
from allennlp.service.db import DemoDatabase, PostgresDemoDatabase
from allennlp.service.permalinks import int_to_slug, slug_to_int
from allennlp.service.predictors import Predictor, DemoModel

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("SANIC_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def run(port: int, workers: int,
        trained_models: Dict[str, DemoModel],
        static_dir: str = None) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))

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

    app.run(port=port, host="0.0.0.0", workers=workers)

def make_app(build_dir: str = None, demo_db: Optional[DemoDatabase] = None) -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name
    start_time = datetime.now(pytz.utc)
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    if build_dir is None:
        # Need path to static assets to be relative to this file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        build_dir = os.path.join(dir_path, '../../demo/build')

    if not os.path.exists(build_dir):
        logger.error("app directory %s does not exist, aborting", build_dir)
        sys.exit(-1)

    app.predictors = {}

    try:
        cache_size = int(CACHE_SIZE)  # type: ignore
    except ValueError:
        logger.warning("unable to parse cache size %s as int, disabling cache", CACHE_SIZE)
        cache_size = 0

    @lru_cache(maxsize=cache_size)
    def _caching_prediction(model: Predictor, data: str) -> JsonDict:
        """
        Just a wrapper around ``model.predict_json`` that allows us to use a cache decorator.
        """
        return model.predict_json(json.loads(data))

    @app.route('/permadata', methods=['POST', 'OPTIONS'])
    async def permadata(req: request.Request) -> response.HTTPResponse: # pylint: disable=unused-variable
        """
        If the user requests a permalink, the front end will POST here with the payload
            { slug: slug }
        which we convert to an integer id and use to retrieve saved results from the database.
        """
        # This is just CORS boilerplate.
        if req.method == "OPTIONS":
            return response.text("")

        # If we don't have a database configured, there are no permalinks.
        if demo_db is None:
            raise ServerError('Permalinks are not enabled', 400)

        # Convert the provided slug to an integer id.
        slug = req.json["slug"]
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

        return response.json({
                "modelName": permadata.model_name,
                "requestData": permadata.request_data,
                "responseData": permadata.response_data
        })

    @app.route('/predict/<model_name>', methods=['POST', 'OPTIONS'])
    async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if req.method == "OPTIONS":
            return response.text("")

        model = app.predictors.get(model_name.lower())
        if model is None:
            raise ServerError("unknown model: {}".format(model_name), status_code=400)

        data = req.json
        log_blob = {"model": model_name, "inputs": data, "cached": False, "outputs": {}}

        # See if we hit or not. In theory this could result in false positives.
        pre_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        try:
            if cache_size > 0:
                # lru_cache insists that all function arguments be hashable,
                # so unfortunately we have to stringify the data.
                prediction = _caching_prediction(model, json.dumps(data))
            else:
                # if cache_size is 0, skip caching altogether
                prediction = model.predict_json(data)
        except KeyError as err:
            raise ServerError("Required JSON field not found: " + err.args[0], status_code=400)

        post_hits = _caching_prediction.cache_info().hits  # pylint: disable=no-value-for-parameter

        # Add to database and get permalink
        if demo_db is not None:
            try:
                perma_id = demo_db.add_result(headers=req.headers,
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

        if post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            log_blob["cached"] = True
            await asyncio.sleep(0.25)

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

        logger.info("prediction: %s", json.dumps(log_blob))

        return response.json(prediction)

    @app.route('/models')
    async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """list the available models"""
        return response.json({"models": list(app.predictors.keys())})

    @app.route('/info')
    async def info(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """List metadata about the running webserver"""
        uptime = str(datetime.now(pytz.utc) - start_time)
        git_version = os.environ.get('SOURCE_COMMIT') or ""
        return response.json({
                "start_time": start_time_str,
                "uptime": uptime,
                "git_version": git_version,
                "githubUrl": "http://github.com/allenai/allennlp/commit/" + git_version})

    # As a SPA, we need to return index.html for /model-name and /model-name/permalink
    @app.route('/semantic-role-labeling')
    @app.route('/machine-comprehension')
    @app.route('/textual-entailment')
    @app.route('/coreference-resolution')
    @app.route('/named-entity-recognition')
    @app.route('/semantic-role-labeling/<permalink>')
    @app.route('/machine-comprehension/<permalink>')
    @app.route('/textual-entailment/<permalink>')
    @app.route('/coreference-resolution/<permalink>')
    @app.route('/named-entity-recognition/<permalink>')
    async def return_page(req: request.Request, permalink: str = None) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """return the page"""
        return await response.file(os.path.join(build_dir, 'index.html'))

    app.static('/', os.path.join(build_dir, 'index.html'))
    app.static('/', build_dir)

    return app
