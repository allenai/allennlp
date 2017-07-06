# Service

This service uses Flask to serve AllenNLP models through a REST API.
(Or it will once they're ready; right now it serves one-line placeholder models.)

Right now the API has two routes.

`GET /models` returns a list of the available models.

`POST /predict/<model_name>` asks the specified model for a prediction, based on the data in the request body.
The current placeholder models expect a field called `input` whose value is a string.
They return the same input, along with the model name and an `output` field that is also a string.

It also serves a bare-bones web page that provides a front-end for these predictions.
