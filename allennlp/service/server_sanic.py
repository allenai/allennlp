"""
A `Sanic <http://sanic.readthedocs.io/en/latest/>`_ server that serves up
AllenNLP models as well as our demo.

Usually you would use :mod:`~allennlp.commands.serve`
rather than instantiating an ``app`` yourself.
"""
from typing import Dict
import asyncio
import json
import logging
import os
from functools import lru_cache

from sanic import Sanic, response, request
from sanic.exceptions import ServerError

from allennlp.common.util import JsonDict
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("SANIC_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def run(port: int, workers: int, config: Dict[str, str]) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))
    app = make_app()
    app.predictors = {
            name: Predictor.from_archive(load_archive(archive_file))
            for name, archive_file in config.items()
    }
    app.run(port=port, host="0.0.0.0", workers=workers)

def make_app() -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name

    app.static('/', './allennlp/service/static/')
    app.static('/', './allennlp/service/static/index.html')
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

    @app.route('/predict/<model_name>', methods=['POST'])
    async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        model = app.predictors.get(model_name.lower())
        if model is None:
            raise ServerError("unknown model: {}".format(model_name), status_code=400)

        data = req.json

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

        if post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            logger.info("cache hit: %s %s", model_name, json.dumps(data))
            await asyncio.sleep(0.25)

        return response.json(prediction)

    @app.route('/models')
    async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """list the available models"""
        return response.json({"models": list(app.predictors.keys())})

    return app
