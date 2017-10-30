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
import sys
from functools import lru_cache

from sanic import Sanic, response, request
from sanic.exceptions import ServerError

from allennlp.common.util import JsonDict
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

# Can override cache size with an environment variable. If it's 0 then disable caching altogether.
CACHE_SIZE = os.environ.get("SANIC_CACHE_SIZE") or 128

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def run(port: int, workers: int,
        trained_models: Dict[str, str],
        static_dir: str = None) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))

    app = make_app(static_dir)

    for predictor_name, archive_file in trained_models.items():
        archive = load_archive(archive_file)
        predictor = Predictor.from_archive(archive, predictor_name)
        app.predictors[predictor_name] = predictor

    app.run(port=port, host="0.0.0.0", workers=workers)

def make_app(build_dir: str = None) -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name

    if build_dir is None:
        # Need path to static assets to be relative to this file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        build_dir = os.path.join(dir_path, '../../demo/build')

    if not os.path.exists(build_dir):
        logger.error("app directory %s does not exist, aborting", build_dir)
        sys.exit(-1)

    app.static('/', os.path.join(build_dir, 'index.html'))
    app.static('/', build_dir)
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

    @app.route('/permalink/<slug>', methods=['GET'])
    async def permalink(req: request.Request, slug: str) -> response.HTTPResponse: # pylint: disable=unused-argument,unused-variable
        """
        Just return the index.html page
        """
        return await response.file(os.path.join(build_dir, 'index.html'))

    @app.route('/permadata', methods=['POST'])
    async def permadata(req: request.Request) -> response.HTTPResponse: # pylint: disable=unused-variable
        """
        Just return the index.html page
        """
        slug = req.json["slug"]

        # data = get_data_for(slug)
        request_data = {"sentence":"If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"}
        response_data = {"words":["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"],"verbs":[{"verb":"liked","description":"If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"were","description":"If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","O","O","O","B-V","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"playing","description":"If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last night] , you will absolutely love what we 're playing tomorrow !","tags":["O","O","O","B-ARG1","I-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","I-ARGM-TMP","O","O","O","O","O","O","O","O","O","O","O"]},{"verb":"will","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [V: will] [ARG1: absolutely love what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},{"verb":"love","description":"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !","tags":["B-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","I-ARGM-ADV","O","B-ARG0","B-ARGM-MOD","B-ARGM-ADV","B-V","B-ARG1","I-ARG1","I-ARG1","I-ARG1","I-ARG1","O"]},{"verb":"'re","description":"If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-V","O","O","O"]},{"verb":"playing","description":"If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !","tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-ARG1","B-ARG0","O","B-V","B-ARGM-TMP","O"]}],"tokens":["If","you","liked","the","music","we","were","playing","last","night",",","you","will","absolutely","love","what","we","'re","playing","tomorrow","!"]}

        return request.json({
                "request_data": request_data,
                "response_data": response_data
        })

    @app.route('/predict/<model_name>', methods=['POST'])
    async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
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

        if post_hits > pre_hits:
            # Cache hit, so insert an artifical pause
            log_blob["cached"] = True
            await asyncio.sleep(0.25)

        # The model predictions are extremely verbose, so we only log the most human-readable
        # parts of them.
        if model_name == "machine-comprehension":
            log_blob["outputs"]["best_span_str"] = prediction["best_span_str"]
        elif model_name == "textual-entailment":
            log_blob["outputs"]["label_probs"] = prediction["label_probs"]
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

    return app
