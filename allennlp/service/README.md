# Service

This service serves AllenNLP models through a REST API.

To start the server run the following command.

```bash
python -m allennlp.run serve --port 8000
```

Right now the API has two routes.

`GET /models` returns a list of the available models.

`POST /predict/<model_name>` asks the specified model for a prediction, based on the data in the request body. Each model expects different inputs:

* Semantic Role Labeling: `{"sentence": "..."}`
* Bidaf (question answering): `{"paragraph": "...", "question": "..."}`
* Snli: `{"premise": "...", "hypothesis": "..."}`

It also serves a web demo for each of these at `/`,
that's probably how you want to use these.
