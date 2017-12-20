"""
A `Flask <http://flask.pocoo.org/>`_ server for serving predictions
from a single AllenNLP model. It also includes a very, very bare-bones
web front-end for exploring predictions (or you can provide your own).
"""
from typing import List, Callable
import json
import logging
import os
from string import Template
import sys

from flask import Flask, request, Response, jsonify, send_file, send_from_directory
from flask_cors import CORS

from allennlp.common import JsonDict
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
from allennlp.service.server_flask import ServerError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_app(predictor: Predictor,
             field_names: List[str] = None,
             static_dir: str = None,
             sanitizer: Callable[[JsonDict], JsonDict] = None,
             use_cors: bool = False) -> Flask:
    """
    Creates a Flask app that serves up the provided ``Predictor``
    along with a front-end for interacting with it.

    If you want to use the built-in bare-bones HTML, you must provide the
    field names for the inputs (which will be used both as labels
    and as the keys in the JSON that gets sent to the predictor).

    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``. In that case you
    don't need to supply the field names -- that information should
    be implicit in your demo site.

    In addition, if you want somehow transform the JSON prediction
    (e.g. by removing probabilities or logits)
    you can do that by passing in a ``sanitizer`` function.
    """

    if static_dir is not None and not os.path.exists(static_dir):
        logger.error("app directory %s does not exist, aborting", static_dir)
        sys.exit(-1)
    elif static_dir is None and field_names is None:
        logger.error("must specify either build_dir or field_names")
        sys.exit(-1)

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        if static_dir is not None:
            return send_file(os.path.join(static_dir, 'index.html'))
        else:
            html = _html(field_names)
            return Response(response=html, status=200)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()

        prediction = predictor.predict_json(data)
        if sanitizer is not None:
            prediction = sanitizer(prediction)

        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))

        return jsonify(prediction)

    @app.route('/<path:path>')
    def static_proxy(path: str) -> Response: # pylint: disable=unused-variable
        if static_dir is not None:
            return send_from_directory(static_dir, path)
        else:
            raise ServerError("static_dir not specified", 404)

    if use_cors:
        return CORS(app)
    else:
        return app


if __name__ == "__main__":
    # Executing this file runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this (except maybe to test changes to the stock HTML), but this shows
    # you what you'd do in your own code to run your own demo.
    main()


def main():
    # Make sure all the classes you need for your Model / Predictor / DatasetReader / etc...
    # are imported here, because otherwise they can't be constructed ``from_params``.

    archive = load_archive('tests/fixtures/bidaf/serialization/model.tar.gz')
    predictor = Predictor.from_archive(archive, 'machine-comprehension')

    def sanitizer(prediction: JsonDict) -> JsonDict:
        """
        Only want best_span results.
        """
        return {key: value
                for key, value in prediction.items()
                if key.startswith("best_span")}

    app = make_app(predictor=predictor,
                   field_names=['passage', 'question'],
                   sanitizer=sanitizer)

    app.run(port=8888, host="0.0.0.0")

#
# HTML and Templates for the default bare-bones app are below
#

_PAGE_TEMPLATE = Template("""
<html>
    <body>
        $inputs
        <button onclick="predict()">Predict</button>
        <div id="output"></div>
    </body>
    <script>
    function predict() {
        var quotedFieldList = $qfl;
        var data = {};
        quotedFieldList.forEach(function(fieldName) {
            data[fieldName] = document.getElementById("input-" + fieldName).value;
        })

        var xhr = new XMLHttpRequest();
        xhr.open('POST', '/predict');
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function() {
            if (xhr.status == 200) {
                document.getElementById("output").innerHTML = xhr.responseText;
            }
        };
        xhr.send(JSON.stringify(data));
    }
    </script>
</html>
""")

_SINGLE_INPUT_TEMPLATE = Template("""
        $field_name
        <br>
        <input type="text" id="input-$field_name" name="$field_name" value="temp">
        <br>
""")

def _html(field_names: List[str]) -> str:
    """
    Returns bare bones HTML for serving up an input form with the
    specified fields that can render predictions from the configured model.
    """
    inputs = ''.join(_SINGLE_INPUT_TEMPLATE.substitute(field_name=field_name)
                     for field_name in field_names)

    quoted_field_names = [f"'{field_name}'" for field_name in field_names]
    quoted_field_list = f"[{','.join(quoted_field_names)}]"

    return _PAGE_TEMPLATE.substitute(inputs=inputs, qfl=quoted_field_list)
