# Service

This service serves AllenNLP models through a REST API.
(Or it will once they're ready; right now it serves one-line placeholder models.)

Right now we are considering both Flask and Sanic. Accordingly, there are two
versions of the server.

To start the flask version run

```bash
FLASK_APP=allennlp/service/server_flask.py flask run --port 5001
```

(substitute whatever port you want).

To start the sanic version run

```bash
python -m sanic --workers 4 --port 5001 allennlp.service.server_sanic.app
```

Right now the API has two routes.

`GET /models` returns a list of the available models.

`POST /predict/<model_name>` asks the specified model for a prediction, based on the data in the request body.
The current placeholder models expect a field called `input` whose value is a string.
They return the same input, along with the model name and an `output` field that is also a string.

It also serves a bare-bones web page that provides a front-end for these predictions.
