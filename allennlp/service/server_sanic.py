from allennlp.service.predictors import PredictorCollection, sanitize

from sanic import Sanic, response, request
from sanic.exceptions import ServerError

def run(port: int) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))
    app = make_app()
    # TODO(joelgrus): make this configurable
    app.predictors = PredictorCollection.default()
    app.run(port=port, host="0.0.0.0")

def make_app() -> Sanic:
    app = Sanic(__name__)  # pylint: disable=invalid-name

    app.static('/', './allennlp/service/static/')
    app.static('/', './allennlp/service/static/index.html')
    app.predictors = PredictorCollection()

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

        sanitized = sanitize(prediction)
        return response.json(sanitized)

    @app.route('/models')
    async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument, unused-variable
        """list the available models"""
        return response.json({"models": app.predictors.list_available()})

    return app
