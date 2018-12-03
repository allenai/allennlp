"""
This is a tiny webapp for generating configuration stubs for your models.
It's still experimental.

```
python -m allennlp.service.config_explorer
```

will launch the app on `localhost:8123` (you can specify a different port if you like).

It can also incorporate your own classes if you use the `include_package` flag:

```
python -m allennlp.service.config_explorer \
    --include-package my_library
```
"""
# pylint: disable=too-many-return-statements
from typing import Sequence
import logging

from flask import Flask, request, Response, jsonify, send_file

from allennlp.common.configuration import configure, choices
from allennlp.common.util import import_submodules
from allennlp.service.server_simple import ServerError

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
    def index() -> Response:  # pylint: disable=unused-variable
        return send_file('config_explorer.html')

    @app.route('/api/config/')
    def api_config() -> Response:  # pylint: disable=unused-variable
        """
        There are basically two things that can happen here.
        If this method is called with a ``Registrable`` class (e.g. ``Model``),
        it should return the list of possible ``Model`` subclasses.
        If it is called with an instantiable subclass (e.g. ``CrfTagger``),
        is should return the config for that subclass.

        This is complicated by the fact that some Registrable base classes
        (e.g. Vocabulary, Trainer) are _themselves_ instantiable.

        We handle this in two ways: first, we insist that the first case
        include an extra ``get_choices`` parameter. That is, if you call
        this method for ``Trainer`` with get_choices=true, you get the list
        of Trainer subclasses. If you call it without that extra flag, you
        get the config for the class itself.

        There are basically two UX situations in which this API is called.
        The first is when you have a dropdown list of choices (e.g. Model types)
        and you select one. Such an API request is made *without* the get_choices flag,
        which means that the config is returned *even if the class in question
        is a Registrable class that has subclass choices*.

        The second is when you click a "Configure" button, which configures
        a class that may (e.g. ``Model``) or may not (e.g. ``FeedForward``)
        have registrable subclasses. In this case the API request is made
        with the "get_choices" flag, but will return the corresponding config
        object if no choices are available (e.g. in the ``FeedForward``) case.

        This is not elegant, but it works.
        """
        class_name = request.args.get('class', '')
        get_choices = request.args.get('get_choices', None)

        # Get the configuration for this class name
        config = configure(class_name)
        try:
            # May not have choices
            choice5 = choices(class_name)
        except ValueError:
            choice5 = []

        if get_choices and choice5:
            return jsonify({
                    "className": class_name,
                    "choices": choice5
            })
        else:
            return jsonify({
                    "className": class_name,
                    "config": config.to_json()
            })

    return app
