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
             title: str = "AllenNLP Demo",
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
            html = _html(title, field_names)
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


def main():
    # Executing this file runs the simple service with the bidaf test fixture
    # and the machine-comprehension predictor. There's no good reason you'd want
    # to do this (except maybe to test changes to the stock HTML), but this shows
    # you what you'd do in your own code to run your own demo.

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
    <head>
        <title>
            $title
        </title>
        <style>
            $css
        </style>
    </head>
    <body>
        <div class="pane-container">
            <div class="pane model">
                <div class="pane__left model__input">
                    <div class="model__content">
                        <h2><span>$title</span></h2>
                        <div class="model__content">
                            $inputs
                            <div class="form__field form__field--btn">
                                <button type="button" class="btn btn--icon-disclosure" onclick="predict()">Predict</button>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="pane__right model__output model__output--empty">
                    <div class="pane__thumb"></div>
                    <div class="model__content">
                        <div id="output" class="output">
                            <div class="placeholder">
                                <div class="placeholder__content">
                                    <p>Run model to view results</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
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
                document.getElementById("output").innerHTML = (
                    "<pre>" +
                    JSON.stringify(JSON.parse(xhr.responseText), null, 2) +
                    "</pre>"
                );
            }
        };
        xhr.send(JSON.stringify(data));
    }
    </script>
</html>
""")

_SINGLE_INPUT_TEMPLATE = Template("""
        <div class="form__field">
            <label for="input-$field_name">$field_name</label>
            <input type="text" id="input-$field_name" type="text" required value placeholder="input goes here">
        </div>
""")


# TODO(joelgrus): Figure out which of this CSS is unnecessary
_CSS = """
/*********************************
Unminified compiled Less output
*********************************/

.c-bg-slate {
  background: #162328
}

.c-bg-white {
  background: #fff
}

.c-bg-gray-light {
  background: #f6f8fa
}

.c-bg-blue {
  background: #2085bc
}

.c-bg-blue-light {
  background: #40affd
}

.c-bg-blue-med {
  background: #309add
}

.c-off-black {
  background: #111212
}

.u-mp0 {
  margin: 0;
  padding: 0
}

.u-100 {
  width: 100%;
  height: 100%
}

.u-w100 {
  width: 100%
}

.u-h100 {
  height: 100%
}

.u-tl {
  top: 0;
  left: 0
}

.u-pe {
  content: ""
}

.u-child100 {
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0
}

.u-pe100 {
  display: block;
  position: absolute;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  content: ""
}

.u-blocklist {
  list-style: none;
  display: block;
  margin: 0;
  padding: 0
}

.u-hidden {
  display: none
}

.u-bg-clip {
  background-clip: padding-box
}

.u-select-none {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-touch-callout: none
}

.u-appearance-none {
  -moz-appearance: none;
  -webkit-appearance: none
}

.u-disable-touch-callout {
  -webkit-touch-callout: none
}

.u-disable-tap-hilight {
  -webkit-tap-highlight-color: transparent;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-touch-callout: none
}

.u-nowrap {
  white-space: nowrap
}

body,
html {
  min-width: 48em;
  background: #f9fafc;
  font-size: 16px
}

.font-size-reset {
  font-size: 1em
}

.constrained {
  max-width: 80em;
  margin: auto
}

.constrained--med {
  max-width: 65em;
  padding: 1.25em 0 3.75em
}

.constrained--sm {
  max-width: 50em;
  padding: 1.25em 0 3.75em
}

.constrained--narrow {
  max-width: 47.5em
}

* {
  font-family: sans-serif;
  color: #232323
}

.constrained h1 {
  font-weight: 200;
  font-size: 2.5em
}

section {
  background: #fff
}

#root,
#root > div {
  width: 100%;
  height: 100%;
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  -ms-flex-direction: column;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  flex-direction: column
}

code,
code span,
pre,
.output {
  font-family: 'Roboto Mono', monospace!important
}

code {
  background: #f6f8fa
}

pre>code {
  font-size: .8125em;
  line-height: 1.5em;
  padding: 1.875em;
  display: block;
  color: #333;
  overflow-x: auto
}

ol,
ul {
  font-size: 16px
}

li,
p,
td,
th {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-size: 1.125em;
  line-height: 1.5em;
  margin: 1.2em 0
}

p code {
  font-size: .78125em;
  padding: .125em .375em
}

pre {
  margin: 2em 0
}

h1,
h2 {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-weight: 300
}

h4,
h5,
h6 {
  font-size: 1em;
  margin: 1.2em 0
}

h2 {
  font-size: 2em;
  color: rgba(35, 35, 35, .75)
}

.more,
a {
  text-decoration: none;
  cursor: pointer;
  color: #2085bc;
  font-weight: 700;
  -webkit-transition: color .2s ease, opacity .2s ease;
  transition: color .2s ease, opacity .2s ease
}

.more:hover,
a:hover {
  color: #40affd
}

.more:active,
a:active {
  opacity: .66;
  color: #2085bc;
  -webkit-transition-duration: 0s;
  transition-duration: 0s
}

blockquote {
  border-left: #f6f8fa solid .375em;
  padding-left: 1.125em
}

blockquote * {
  color: rgba(35, 35, 35, .75)
}

img {
  max-width: 100%
}

hr {
  display: block;
  border: none;
  height: .375em;
  background: #f6f8fa
}

blockquote,
hr {
  margin: 2.4em 0
}

table {
  width: 100%;
  border-collapse: collapse;
  border: none
}

tr:nth-child(even) {
  background: rgba(184, 198, 213, .13)
}

td,
th {
  text-align: left;
  padding: 1.25em
}

td sub,
th sub {
  vertical-align: baseline;
  padding-left: .375em;
  opacity: .5;
  font-size: 75%
}

th {
  border-bottom: .375em solid rgba(184, 198, 213, .13)
}

.tr--featured td,
.tr--featured td * {
  color: #2085bc
}

.tr--featured td sub {
  opacity: .66
}

#mask {
  width: 100%;
  height: 100%;
  position: fixed;
  top: 0;
  left: 0;
  background: url(../assets/mask.png) center top no-repeat;
  z-index: 999999
}

.btn {
  text-decoration: none;
  cursor: pointer;
  text-transform: uppercase;
  font-size: 1em;
  margin: 0;
  -moz-appearance: none;
  -webkit-appearance: none;
  border: none;
  color: #fff!important;
  display: block;
  background: #2085bc;
  padding: .9375em 3.625em;
  -webkit-transition: background-color .2s ease, opacity .2s ease;
  transition: background-color .2s ease, opacity .2s ease
}

.btn.btn--blue {
  background: #2085bc
}

.btn:focus,
.btn:hover {
  background: #40affd;
  outline: 0
}

.btn:focus {
  box-shadow: 0 0 1.25em rgba(50, 50, 150, .05)
}

.btn:active {
  opacity: .66;
  background: #2085bc;
  -webkit-transition-duration: 0s;
  transition-duration: 0s
}

.btn.btn--icon-disclosure svg {
  width: .4375em;
  height: .75em;
  fill: #fff;
  opacity: .66;
  margin: 0 -.75em 0 .75em
}

.btn:disabled,
.btn:disabled:active,
.btn:disabled:hover {
  cursor: default;
  background: #d0dae3
}

.flex-container {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex
}

.flex-none {
  -webkit-flex: none;
  -ms-flex: none;
  -flex: none
}

.flex-container-row {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row
}

.flex-container-column {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  -ms-flex-direction: column;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  flex-direction: column
}

.flex-container-align-left {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row;
  -webkit-justify-content: flex-start;
  -ms-justify-content: flex-start;
  -webkit-box-pack: start;
  -ms-flex-pack: start;
  justify-content: flex-start
}

.flex-container-align-right {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row;
  -webkit-justify-content: flex-end;
  -ms-justify-content: flex-end;
  -webkit-box-pack: end;
  -ms-flex-pack: end;
  justify-content: flex-end
}

.flex-container-hcentered {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-justify-content: center;
  -ms-justify-content: center;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center
}

.flex-container-vcentered {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center
}

.flex-container-centered {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center;
  -webkit-justify-content: center;
  -ms-justify-content: center;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center
}

.flex-distribute-vertically {
  -webkit-align-content: space-around;
  -ms-flex-line-pack: distribute;
  align-content: space-around
}

form {
  display: block
}

.form__instructions {
  margin: 4.25em 0 2.0625em;
  -webkit-transition: margin .2s ease;
  transition: margin .2s ease
}

.form__instructions span {
  font-size: 1em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  padding-right: .5em;
  max-width: 9.375em;
  color: #7c7c7c
}

.form__instructions select {
  font-size: .8125em;
  max-width: 11.25em
}

.form__field {
  -webkit-transition: margin .2s ease;
  transition: margin .2s ease
}

.form__field+.form__field {
  margin-top: 2.5em
}

.form__field label {
  display: block;
  font-weight: 600;
  font-size: 1.125em
}

.form__field label+* {
  margin-top: 1.25em
}

.form__field input[type=text],
.form__field textarea {
  -moz-appearance: none;
  -webkit-appearance: none;
  width: 100%;
  font-size: 1em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  padding: .8125em 1.125em;
  color: #232323;
  border: .125em solid #d4dce2;
  display: block;
  box-sizing: border-box;
  -webkit-transition: background-color .2s ease, color .2s ease, border-color .2s ease, opacity .2s ease;
  transition: background-color .2s ease, color .2s ease, border-color .2s ease, opacity .2s ease
}

.form__field input[type=text]::-webkit-input-placeholder,
.form__field textarea::-webkit-input-placeholder {
  color: #b4b4b4
}

.form__field input[type=text]:-moz-placeholder,
.form__field textarea:-moz-placeholder {
  color: #b4b4b4
}

.form__field input[type=text]::-moz-placeholder,
.form__field textarea::-moz-placeholder {
  color: #b4b4b4
}

.form__field input[type=text]:-ms-input-placeholder,
.form__field textarea:-ms-input-placeholder {
  color: #b4b4b4
}

.form__field input[type=text]:focus,
.form__field textarea:focus {
  outline: 0;
  border-color: #63a7d4;
  box-shadow: 0 0 1.25em rgba(50, 50, 150, .05)
}

.form__field textarea {
  resize: vertical;
  min-height: 8.25em
}

.form__field .btn {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-touch-callout: none
}

.form__field--btn {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row;
  -webkit-justify-content: flex-end;
  -ms-justify-content: flex-end;
  -webkit-box-pack: end;
  -ms-flex-pack: end;
  justify-content: flex-end
}

@media screen and (max-height:760px) {
  .form__instructions {
    margin: 1.875em 0 1.125em
  }
  .form__field:not(.form__field--btn)+.form__field:not(.form__field--btn) {
    margin-top: 1.25em
  }
}

body,
html {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: 'Source Sans Pro', sans-serif
}

h1 {
  font-weight: 300
}

header {
  width: 100%;
  background: #fff;
  border-bottom: .125em solid #e5eaf0
}

.header__content {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  width: 100%
}

.header__content nav {
  height: 100%;
  margin-left: 1.25em
}

.header__content nav ul,
.header__content nav ul li {
  list-style: none;
  display: block;
  margin: 0;
  padding: 0;
  font-size: 1em
}

.header__content nav ul {
  height: 100%;
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row;
  -webkit-justify-content: flex-end;
  -ms-justify-content: flex-end;
  -webkit-box-pack: end;
  -ms-flex-pack: end;
  justify-content: flex-end;
  margin-right: 1.25em
}

.header__content nav li {
  height: 100%
}

.header__content nav li.nav__home {
  display: none
}

.header__content .nav__link {
  height: 100%;
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center;
  margin: 0;
  text-decoration: none;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  padding: 0 1.25em
}

.header__content .nav__link span {
  font-size: 1em;
  font-size: 1.125em;
  height: 100%;
  -webkit-transition: color .2s ease;
  transition: color .2s ease;
  font-weight: 600;
  color: rgba(28, 47, 58, .5);
  position: relative;
  line-height: 4.625em
}

.header__content .nav__link:hover span {
  color: #1c2f3a
}

.header__content .nav__link:active span {
  color: rgba(28, 47, 58, .4);
  -webkit-transition-duration: 0s;
  transition-duration: 0s
}

.header__content .nav__link--selected,
.header__content .nav__link--selected:active,
.header__content .nav__link--selected:hover {
  background: rgba(255, 255, 255, .13);
  opacity: 1
}

.header__content .nav__link--selected span,
.header__content .nav__link--selected:active span,
.header__content .nav__link--selected:hover span {
  color: #1c2f3a
}

.header__content .nav__link--selected span:after,
.header__content .nav__link--selected:active span:after,
.header__content .nav__link--selected:hover span:after {
  content: "";
  width: 100%;
  height: .375em;
  background: #63a7d4;
  display: block;
  bottom: -30px;
  left: 0;
  position: absolute
}

.header__content__logo {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center;
  margin: 0 2.125em 0 auto;
  padding: 0;
  font-size: 1em
}

.header__content__logo a {
  margin: 0;
  padding: 0;
  display: block
}

header svg {
  fill: #1c2f3a;
  width: 7.75em;
  height: 1.375em
}

#icons {
  top: 0;
  left: 0;
  position: absolute;
  height: 0;
  width: 0;
  visibility: hidden;
  overflow: hidden;
  z-index: -999
}

svg {
  fill: #000
}

.icon__working__gradient__stop {
  stop-color: #40affd
}

.icon__working__gradient__stop--stop4,
.icon__working__gradient__stop--stop5,
.icon__working__gradient__stop--stop6 {
  stop-opacity: 0
}

.icon__working__path {
  fill: url(#icon__working__gradient)
}

.model__output {
  background: #fff
}

.model__output.model__output--empty {
  background: 0 0
}

.placeholder {
  width: 100%;
  height: 100%;
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center;
  -webkit-justify-content: center;
  -ms-justify-content: center;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-touch-callout: none;
  cursor: default
}

.placeholder .placeholder__content {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  -ms-flex-direction: column;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  flex-direction: column;
  -webkit-align-items: center;
  -ms-flex-align: center;
  -webkit-box-align: center;
  align-items: center;
  text-align: center
}

.placeholder svg {
  display: block
}

.placeholder svg.placeholder__empty,
.placeholder svg.placeholder__error {
  width: 6em;
  height: 3.625em;
  fill: #e1e5ea;
  margin-bottom: 2em
}

.placeholder svg.placeholder__error {
  width: 4.4375em;
  height: 4em
}

.placeholder p {
  font-size: 1em;
  margin: 0;
  padding: 0;
  color: #9aa8b2
}

.placeholder svg.placeholder__working {
  width: 3.4375em;
  height: 3.4375em;
  -webkit-animation: working 1s infinite linear;
  animation: working 1s infinite linear
}

@-webkit-keyframes working {
  0% {
    -webkit-transform: rotate(0deg)
  }
  100% {
    -webkit-transform: rotate(360deg)
  }
}

@keyframes working {
  0% {
    -webkit-transform: rotate(0deg);
    -ms-transform: rotate(0deg);
    transform: rotate(0deg)
  }
  100% {
    -webkit-transform: rotate(360deg);
    -ms-transform: rotate(360deg);
    transform: rotate(360deg)
  }
}

.model__content {
  padding: 1.875em 2.5em;
  margin: auto;
  -webkit-transition: padding .2s ease;
  transition: padding .2s ease
}

.model__content:not(.model__content--srl-output) {
  max-width: 61.25em
}

.model__content h2 {
  margin: 0;
  padding: 0;
  font-size: 1em
}

.model__content h2 span {
  font-size: 2em;
  color: rgba(35, 35, 35, .75)
}

.model__content h2 .tooltip,
.model__content h2 span {
  vertical-align: top
}

.model__content h2 span+.tooltip {
  margin-left: .4375em
}

.model__content>h2:first-child {
  margin: -.25em 0 0 -.03125em
}

.model__content__summary {
  font-size: 1em;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  padding: 1.25em;
  background: #f6f8fa
}

@media screen and (min-height:800px) {
  .model__content {
    padding-top: 4.6vh;
    padding-bottom: 4.6vh
  }
}

.pane-container {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  -ms-flex-direction: column;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  flex-direction: column;
  height: 100%
}

.pane {
  display: -webkit-box;
  display: -ms-flexbox;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: row;
  -ms-flex-direction: row;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  flex-direction: row;
  position: relative;
  -webkit-box-flex: 2;
  -webkit-flex: 2;
  -ms-flex: 2;
  flex: 2;
  height: auto;
  min-height: 100%;
  min-height: 34.375em
}

.pane__left,
.pane__right {
  width: 100%;
  height: 100%;
  -webkit-align-self: stretch;
  -ms-flex-item-align: stretch;
  align-self: stretch;
  min-width: 24em;
  min-height: 34.375em
}

.pane__left {
  height: auto;
  min-height: 100%
}

.pane__right {
  width: 100%;
  overflow: auto;
  height: auto;
  min-height: 100%
}

.pane__right .model__content.model__content--srl-output {
  display: inline-block;
  margin: auto
}

.pane__thumb {
  height: auto;
  min-height: 100%;
  margin-left: -.625em;
  position: absolute;
  width: 1.25em
}

.pane__thumb:after {
  display: block;
  position: absolute;
  height: 100%;
  top: 0;
  content: "";
  width: .25em;
  background: #e1e5ea;
  left: .5em
}



.tooltip {
  display: inline-block;
  width: 2.5em;
  height: 2.5em;
  position: relative
}

.tooltip svg.tooltip__trigger {
  position: absolute;
  width: 1.0625em;
  height: 1.0625em;
  top: .875em;
  left: .6875em;
  fill: rgba(28, 47, 58, .18);
  -webkit-transition: fill .2s ease 0s;
  transition: fill .2s ease 0s
}

.tooltip .tooltip__cursor-container {
  position: absolute;
  margin: 2.375em 0 0 -.875em;
  padding-top: .9375em;
  cursor: default;
  z-index: -999;
  opacity: 0;
  -webkit-transition: opacity .2s ease 0s, z-index 0s linear .21s;
  transition: opacity .2s ease 0s, z-index 0s linear .21s
}

.tooltip .tooltip__box {
  -webkit-filter: drop-shadow(0 0 1.25em rgba(28, 47, 58, .3));
  filter: drop-shadow(0 0 1.25em rgba(28, 47, 58, .3));
  width: 30vw;
  max-width: 23.5em;
  background: #1c2f3a;
  padding: 1.5em 1.875em
}

.tooltip .tooltip__box,
.tooltip .tooltip__box * {
  color: #fff;
  fill: #fff
}

.tooltip .tooltip__box p {
  margin: 0;
  padding: 0;
  font-size: 1em;
  -webkit-font-smoothing: subpixel-antialiased;
  -moz-osx-font-smoothing: auto;
  line-height: 1.5em;
  color: rgba(255, 255, 255, .75)
}

.tooltip .tooltip__box p strong {
  color: #fff;
  font-weight: 600
}

.tooltip .tooltip__box:after {
  display: block;
  position: absolute;
  top: 0;
  left: 0;
  content: "";
  width: 0;
  height: 0;
  margin-top: .25em;
  margin-left: 1.25em;
  border-left: .875em solid transparent;
  border-right: .875em solid transparent;
  border-bottom: .875em solid #1c2f3a
}

.tooltip:hover svg.tooltip__trigger {
  fill: #1c2f3a;
  -webkit-transition: fill .2s ease .125s;
  transition: fill .2s ease .125s
}

.tooltip:hover .tooltip__cursor-container {
  display: block;
  z-index: 99999;
  opacity: 1;
  -webkit-transition: opacity .2s ease .125s, z-index 0s linear .1s;
  transition: opacity .2s ease .125s, z-index 0s linear .1s
}

[data-tooltip] {
  position: relative;
  z-index: 2;
  cursor: pointer;
}

[data-tooltip]:before,
[data-tooltip]:after {
  visibility: hidden;
  -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";
  filter: progid: DXImageTransform.Microsoft.Alpha(Opacity=0);
  opacity: 0;
  pointer-events: none;
}

/* Position tooltip above the element */
[data-tooltip]:before {
  position: absolute;
  bottom: 150%;
  left: 50%;
  margin-bottom: 5px;
  margin-left: -80px;
  padding: 7px;
  width: 160px;
  -webkit-border-radius: 3px;
  -moz-border-radius: 3px;
  border-radius: 3px;
  background-color: #000;
  background-color: hsla(0, 0%, 20%, 0.9);
  color: #fff;
  content: attr(data-tooltip);
  text-align: center;
  font-size: 14px;
  line-height: 1.2;
}

/* Triangle hack to make tooltip look like a speech bubble */
[data-tooltip]:after {
  position: absolute;
  bottom: 150%;
  left: 50%;
  margin-left: -5px;
  width: 0;
  border-top: 5px solid #000;
  border-top: 5px solid hsla(0, 0%, 20%, 0.9);
  border-right: 5px solid transparent;
  border-left: 5px solid transparent;
  content: " ";
  font-size: 0;
  line-height: 0;
}

/* Show tooltip content on hover */
[data-tooltip]:hover:before,
[data-tooltip]:hover:after {
  visibility: visible;
  -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=100)";
  filter: progid: DXImageTransform.Microsoft.Alpha(Opacity=100);
  opacity: 1;
}

.ner-table {
  margin-top: 50px;
}

.t-crisp {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale
}

.t-smooth {
  -webkit-font-smoothing: subpixel-antialiased;
  -moz-osx-font-smoothing: auto
}

.t-kerning-default {
  letter-spacing: normal
}

.t-underline {
  text-decoration: underline
}

.t-no-underline {
  text-decoration: none
}

.t-uppercase {
  text-transform: uppercase
}

.t-case-normal {
  text-transform: none
}

.t-bold {
  font-weight: 700
}

.t-thin {
  font-weight: 100
}

.t-caps {
  font-size: .875em;
  font-weight: 700;
  letter-spacing: .025em;
  text-transform: uppercase
}

.t-link {
  text-decoration: none;
  cursor: pointer
}

.t-truncate {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis
}

.t-center {
  text-align: center
}

.t-left {
  text-align: left
}

.t-right {
  text-align: right
}

.t-white,
.t-white * {
  color: #fff;
  fill: #fff
}

.t-sm {
  font-size: 1em
}

.t-white .t-sm,
.t-white .t-sm * {
  color: rgba(255, 255, 255, .66)
}

.srl__vizualization-types {
  background: #f9fafc;
  display: flex;
  list-style-type: none;
  margin: 0;
  padding: 0 1.25em;
  border-bottom: 1px solid #e5eaf0;
}

.srl__vizualization-types li {
  margin: 0 1em 0 0;
  padding: 1em 1.25em 0.5em;
  border-bottom: 6px solid transparent;
}

.srl__vizualization-types li a {
  color: rgba(28, 47, 58, .5);
}


.srl__vizualization-types .srl__vizualization-types__active-type {
  border-bottom-color: #63a7d4;
}

.srl__vizualization-types .srl__vizualization-types__active-type a {
  color: #1c2f3a;
}

.coref__span {
  background-color: #40affd;
  color: #fff;
}

.hierplane__visualization-verbs {
  background: #f9fafc;
  padding: 1em 1.25em;
  border-bottom: 1px solid #e5eaf0;
  color: rgba(28, 47, 58, .5);
}

.hierplane__visualization-verbs a {
  display: inline-block;
  margin-right: 1em;
}

.hierplane__visualization-verbs a svg {
  fill: rgba(28, 47, 58, .5);
}

.hierplane__visualization-verbs a:hover svg {
  fill: #1c2f3a;
}

.hierplane__visualization-verbs__prev {
  transform: rotateY(-180deg);
}

.hierplane__visualization-verbs__label {
  user-select: none;
}

/* THESE ARE HACKS PUT HERE BY CODEVIKING. */

/*
  Center the controls in the toolbar.
 */
.hierplane__visualization .hierplane .parse-tree-toolbar .parse-tree-toolbar__item .parse-tree-toolbar__item__label {
  top: 50%;
  margin-top: -13px;
}

/*
  Make the toolbar background transparent, as it looks weird when it matches the color of the
  tab / verb-select bars.
 */
.hierplane__visualization .hierplane--theme-light .parse-tree-toolbar {
  background: transparent;
  border: none;
}
.hierplane__visualization .hierplane--theme-light .parse-tree-toolbar:before {
  background: transparent;
  border: none;
}

/*
   This eliminates the negative left margin used to artifically bump punctuation such that it
   doesn't appear to have space before it. This isn't worth the side-effect in SRL, as here
   we have spans that contain things other than punctuation that are ignored. This can in the
   long run potentially be fixed in hierplane...but that might not happen.
*/
.hierplane__visualization .hierplane #passage p .passage__readonly .span-slice__ignored {
  margin-left: 0 !important;
}

/*
  Adjust the "passage" display so that it accomodates the entire sentence, if possible. This
  prevents the user from having to scroll when the sentence is long.
*/
.hierplane__visualization .hierplane--theme-light #passage {
  max-height: initial;
  height: auto !important;
  flex: 1;
}

/*
  Resolve wrapping issues.
 */
.hierplane__visualization .hierplane #passage p span {
  display: inline;
}
"""

def _html(title: str, field_names: List[str]) -> str:
    """
    Returns bare bones HTML for serving up an input form with the
    specified fields that can render predictions from the configured model.
    """
    inputs = ''.join(_SINGLE_INPUT_TEMPLATE.substitute(field_name=field_name)
                     for field_name in field_names)

    quoted_field_names = [f"'{field_name}'" for field_name in field_names]
    quoted_field_list = f"[{','.join(quoted_field_names)}]"

    return _PAGE_TEMPLATE.substitute(title=title,
                                     css=_CSS,
                                     inputs=inputs,
                                     qfl=quoted_field_list)

if __name__ == "__main__":
    main()
