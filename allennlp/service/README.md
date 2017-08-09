# Service

This service serves AllenNLP models through a REST API.

Right now we are considering both Flask and Sanic. Accordingly, there are two
versions of the server. (At some point soon one of them will be
removed, most likely Flask.)

To start the Flask version, from the top level directory run

```bash
allennlp/run serve --backend flask --port 8000
```

(substitute whatever port you want).

To start the sanic version just substitute `sanic` for the backend

```bash
allennlp/run serve --backend sanic --port 8000
```

Right now the API has two routes.

`GET /models` returns a list of the available models.

`POST /predict/<model_name>` asks the specified model for a prediction, based on the data in the request body. Each model expects different inputs:

* Semantic Role Labeling: `{"sentence": "..."}`
* Bidaf (question answering): `{"paragraph": "...", "question": "..."}`
* Snli: `{"premise": "...", "hypothesis": "..."}`

It also serves a web demo for each of these at `/`,
that's probably how you want to use these.
