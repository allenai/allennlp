"""
This is a tiny webapp for generating configuration stubs for your models.
It's very hacky and very experimental, so don't rely on it for anything important.
"""
# pylint: disable=too-many-return-statements
from typing import List, Union, Optional, Dict, Sequence, Tuple
import argparse
import logging
from string import Template
import re
import sys

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common.configuration import configure, Config, ConfigItem, full_name, _NO_DEFAULT, is_configurable
from allennlp.common.util import import_submodules
from allennlp.service.server_flask import ServerError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_app(include_packages: Sequence[str] = ()) -> Flask:
    """
    Creates a Flask app that serves up a simple configuration wizard.
    """
    # Load modules
    for package_name in include_packages:
        import_submodules(package_name)

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        class_name = request.args.get('class', '')

        try:
            config = configure(class_name)
        except:
            # TODO(joelgrus): better error handling
            raise

        if isinstance(config, Config):
            html = config_html(class_name, config)
        else:
            html = choices_html(class_name, config)

        return Response(response=html, status=200)

    return app


def main(args):
    parser = argparse.ArgumentParser(description='Serve up a simple configuration wizard')

    parser.add_argument('--port', type=int, default=8123, help='port to serve the wizard on')

    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')

    args = parser.parse_args(args)

    app = make_app(args.include_package)
    CORS(app)

    http_server = WSGIServer(('0.0.0.0', args.port), app)
    print(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()

#
# HTML and Templates for the default bare-bones app are below
#
_BASE_HTML_TEMPLATE = Template("""
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
        $body
    </body>
</html>
""")

_INDENT = "    "

def _remove_prefix(class_name: str) -> str:
    rgx = r"^(typing\.|builtins\.)"
    return re.sub(rgx, "", class_name)

def _render_annotation(annotation: Optional[type]) -> str:
    # Special case to handle None:
    if annotation is None:
        return "?"

    class_name = _remove_prefix(full_name(annotation))
    origin = getattr(annotation, '__origin__', None)
    args = getattr(annotation, '__args__', ())

    # Special handling for compound types
    if origin == Dict:
        key_type, value_type = args
        return f"""Dict[{_render_annotation(key_type)}, {_render_annotation(value_type)}]"""
    elif origin in (Tuple, List, Sequence):
        return f"""{_remove_prefix(str(origin))}[{", ".join(_render_annotation(arg) for arg in args)}]"""
    elif origin == Union:
        # Special special case to handle optional types:
        if len(args) == 2 and args[-1] == type(None):
            return f"""Optional[{_render_annotation(args[0])}]"""
        else:
            return f"""Union[{", ".join(_render_annotation(arg) for arg in args)}]"""
    elif is_configurable(annotation):
        return f"""<a href="/?class={class_name}">{class_name}</a>"""
    else:
        return class_name


def _render_item(item: ConfigItem) -> str:
    optional = item.default_value != _NO_DEFAULT

    if optional:
        indent = _INDENT + '// '
    else:
        indent = _INDENT


    default = f" (default: {item.default_value} )" if optional else ""
    comment = f" // {item.comment}" if item.comment else ""
    annotation = _render_annotation(item.annotation)

    return f"""{indent}"{item.name}": {annotation}{default}{comment}\n"""

def _render_type(config: Config) -> str:
    if config.typ3:
        return f"""{_INDENT}"type": "{config.typ3}"\n"""
    else:
        return ""

def _api_link(class_name: str) -> str:
    if class_name.startswith("torch."):
        url = f"https://pytorch.org/docs/stable/search.html?q={class_name}&check_keywords=no"
    elif class_name.startswith("allennlp."):
        url = f"https://allenai.github.io/allennlp-docs/search.html?q={class_name}&check_keywords=no"
    else:
        # no url
        return class_name

    return f"""<a href = "{url}" target="_blank">{class_name}</a>"""


def _render_config(config: Config) -> str:
    rendered_type = _render_type(config)
    rendered_items = "".join(_render_item(item) for item in config.items)

    return f"""<div class="config"><pre>\n{{\n{rendered_type}{rendered_items}}}</pre></div>"""

def config_html(class_name: str, config: Config) -> str:
    title = f"Config Explorer: {class_name}" if class_name else "Config Explorer"

    body = f"""
    <div class="label">Stub configuration for {_api_link(class_name) if class_name else "AllenNLP"}</div>
    <div class="config">{_render_config(config)}</div>
    """

    return _BASE_HTML_TEMPLATE.substitute(title=title, css=_CSS, body=body)

def _render_choice(class_name: str) -> str:
    return f"""<li class="choice"><a href="/?class={class_name}">{class_name}</a></li>\n"""


def choices_html(class_name: str, choices: List[str]) -> str:
    title = f"Config Explorer: {class_name}"
    rendered_choices = "".join(_render_choice(choice) for choice in choices)

    body = f"""
    <div class="label">Choose a specific {_api_link(class_name)}:</div>
    <div class="choices">
        <ul>
            {rendered_choices}
        </ul>
    </div>
    """

    return _BASE_HTML_TEMPLATE.substitute(title=title, css=_CSS, body=body)

_CSS = """

"""

if __name__ == "__main__":
    main(sys.argv[1:])
