from allennlp.service.servable import ServableCollection

from sanic import Sanic, response, request
from sanic.config import LOGGING
from sanic.exceptions import ServerError

# Move access.log and error.log to /tmp by default
# If someone really wants them, they can move them back
LOGGING['handlers']['accessTimedRotatingFile']['filename'] = '/tmp/sanic_access.log'
LOGGING['handlers']['errorTimedRotatingFile']['filename'] = '/tmp/sanic_error.log'


app = Sanic(__name__)  # pylint: disable=invalid-name
app.static('/', 'allennlp/service/index.html')
app.static('/index.html', 'allennlp/service/index.html')
app.servables = ServableCollection()

def run(port: int) -> None:
    """Run the server programatically"""
    print("Starting a sanic server on port {}.".format(port))
    # TODO(joelgrus): make this configurable
    app.servables = ServableCollection.default()
    app.run(port=port, host="0.0.0.0")

@app.route('/predict/<model_name>', methods=['POST'])
async def predict(req: request.Request, model_name: str) -> response.HTTPResponse:
    """make a prediction using the specified model and return the results"""
    model = app.servables.get(model_name.lower())
    if model is None:
        raise ServerError("unknown model: {}".format(model_name), status_code=400)

    # TODO(joelgrus): error handling
    data = req.json
    prediction = model.predict_json(data)

    return response.json(prediction)

@app.route('/models')
async def list_models(req: request.Request) -> response.HTTPResponse:  # pylint: disable=unused-argument
    """list the available models"""
    return response.json({"models": app.servables.list_available()})
