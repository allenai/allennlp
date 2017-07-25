from allennlp.service.servable import ServableCollection

from sanic import Sanic, response, request
from sanic.exceptions import ServerError

# TODO(joelgrus): make this configurable
servables = ServableCollection.default()  # pylint: disable=invalid-name

app = Sanic(__name__)  # pylint: disable=invalid-name
app.static('/', 'allennlp/service/index.html')
app.static('/index.html', 'allennlp/service/index.html')

@app.route('/predict/<model_name>', methods=['POST'])
async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:
    """make a prediction using the specified model and return the results"""
    model = servables.get(model_name.lower())
    if model is None:
        raise ServerError("unknown model: {}".format(model_name), status_code=400)

    # TODO(joelgrus): error handling
    data = req.json
    prediction = model.predict_json(data)

    return response.json(prediction)

@app.route('/models')
async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument
    """list the available models"""
    return response.json({"models": servables.list_available()})
