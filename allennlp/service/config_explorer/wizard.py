"""
A `Flask <http://flask.pocoo.org/>`_ server for
running a fun little configuration wizard.
"""
from typing import List, Dict
import importlib
import inspect
import json
import logging
import os
from string import Template

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from gevent.wsgi import WSGIServer

from allennlp.common import JsonDict
from allennlp.common.registrable import Registrable
from allennlp.service.config_explorer.introspection import import_all_submodules, collect_classes, full_path, get_infos
from allennlp.service.server_flask import ServerError


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def make_app(additional_modules: List[str] = None, use_cors: bool = False) -> Flask:
    module_names = ['allennlp'] + (additional_modules or [])

    registered_classes = {}
    for module_name in module_names:
        import_all_submodules(module_name)
        module = importlib.import_module(module_name)
        registered_classes.update(collect_classes(module))
    registry = Registrable._registry  # pylint: disable=protected-access

    app = Flask(__name__)

    # Mapping: className -> [acceptable values]
    types = {
            full_path(clas): {name: full_path(subclass)
                              for name, subclass in inner_dict.items()}
            for clas, inner_dict in registry.items()
    }

    # Add registered classes for Pytorch imports
    for base_name, sub_dict in types.items():
        for sub_name, sub in sub_dict.items():
            if sub not in registered_classes:
                print("-->", sub)

    # Mapping: className -> [parameter_infos]
    infos = get_infos(registered_classes)

    with open(os.path.join(os.path.dirname(__file__), 'index.html')) as f:
        template = Template(f.read())
    html = template.substitute(types=json.dumps(types), infos=json.dumps(infos))

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route('/')
    def index() -> Response: # pylint: disable=unused-variable
        return Response(response=html, status=200)

    @app.route('/subclass', methods=['POST', 'OPTIONS'])
    def subclass() -> Response:  # pylint: disable=unused-variable
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        class_name = data['className']
        clas = registered_classes[class_name].clas
        typ = data['type']

        subclass = registry[clas][typ]
        subclass_name = full_path(subclass)
        registered_subclass = registered_classes[subclass_name]
        signature = registered_subclass.signature
        params = signature.parameters

        config_infos: List[JsonDict] = []

        for param in params.values():
            param_name = param.name
            if param_name == 'self':
                continue

            default = param.default
            annotation = param.annotation
            optional = (default != inspect._empty)

            info: JsonDict = {
                'name': param_name,
                'default': default,
                'optional': optional,
                'annotation': get_info(registered_classes, annotation)
            }

            config_infos.append(info)

        return jsonify(config_infos)

    if use_cors:
        return CORS(app)
    else:
        return app


def main():
    app = make_app()
    http_server = WSGIServer(('0.0.0.0', 8888), app)
    http_server.serve_forever()

if __name__ == "__main__":
    main()
