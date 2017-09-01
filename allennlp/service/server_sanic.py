"""
A `Sanic <http://sanic.readthedocs.io/en/latest/>`_ server that serves up
AllenNLP models as well as our demo.

Usually you would use :mod:`~allennlp.commands.serve`
rather than instantiating an ``app`` yourself.
"""
from typing import Dict
from sanic import Sanic, response, request
from sanic.exceptions import ServerError

from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

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

    @app.route('/predict/<model_name>', methods=['POST'])
    async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        model = app.predictors.get(model_name.lower())
        if model is None:
            raise ServerError("unknown model: {}".format(model_name), status_code=400)

        data = req.json

        try:
            prediction = model.predict_json(data)
        except KeyError as err:
            raise ServerError("Required JSON field not found: " + err.args[0], status_code=400)

        return response.json(prediction)

    @app.route('/models')
    async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """list the available models"""
        return response.json({"models": list(app.predictors.keys())})

    return app
