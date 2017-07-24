from allennlp.service.models import models

from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_url_path='')  # pylint: disable=invalid-name


@app.route('/')
def root() -> Response:
    """serve index.html at the root"""
    return send_from_directory('.', 'index.html')


class UnknownModel(Exception):
    """can't get a prediction from a model we don't know about"""
    def __init__(self, model_name: str) -> None:
        Exception.__init__(self)
        self.model_name = model_name


@app.errorhandler(UnknownModel)
def handle_unknown_model(error: UnknownModel) -> Response:
    """return a 400 Bad Request error for unknown models"""
    response = jsonify({"message": "unknown model", "model_name": error.model_name})
    response.status_code = 400
    return response


@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name: str) -> Response:
    """make a prediction using the specified model and return the results"""
    model_fn = models.get(model_name.lower())
    if model_fn is None:
        raise UnknownModel(model_name)
    model = model_fn()

    # TODO(joelgrus): error handling
    data = request.get_json()
    prediction = model(data)

    return jsonify(prediction)


@app.route('/models')
def list_models() -> Response:
    """list the available models"""
    return jsonify({"models": list(models.keys())})
