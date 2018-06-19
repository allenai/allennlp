"""
This is a tiny webapp for generating configuration stubs for your models.
It's very hacky and very experimental, so don't rely on it for anything important.

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
import argparse
import logging
import sys

from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from allennlp.common.configuration import configure, Config
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
        return Response(response=_HTML, status=200)

    @app.route('/debug/')
    def wizard() -> Response:  # pylint: disable=unused-variable
        """
        This is just for quick iteration while developing, don't use it otherwise.
        """
        return send_file('config_explorer.html')

    @app.route('/api/config/')
    def api() -> Response:  # pylint: disable=unused-variable
        class_name = request.args.get('class', '')

        config = configure(class_name)

        if isinstance(config, Config):
            return jsonify({
                    "className": class_name,
                    "config": config.to_json()
            })
        else:
            return jsonify({
                    "className": class_name,
                    "choices": config
            })

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
    print(f"serving Config Explorer on port {args.port}")
    http_server.serve_forever()

_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>AllenNLP Configuration Wizard (alpha)</title>
    <style>
        div {
            display: table;
        }

        * {
            font-family: sans-serif;
        }

        h1,
        h2 {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-weight: 300
        }

        h2 {
            font-size: 2em;
        }

        .optional {
            color: gray;
        }

        .required {
            color: black;
        }

        span.name {
            font-weight: bold;
            margin: 5px;
        }

        .required.incomplete > span.name {
            background-color: lightcoral;
        }

        .annotation {
            font-size: 90%;
            margin-left: 10px;
            margin-right: 10px;
            color: #2085bc;
        }

        .prefix {
            font-size: 75%;
            margin-left: 10px;
        }


        .default-value {
            color: #979a9d;
            font-size: 90%;
        }

        div#rendered-json {
            margin: 10px;
            border: 1px solid black;
            font-family: monospace;
            white-space: pre;
        }

        .config, .list, .dict {
            margin-top: 5px;
            margin-left: 40px;
        }

        .choices-dropdown {
            margin-left: 20px;
        }

        .config-item {
            margin-top: 2px;
        }

        .dict-item {
            border: 1px dotted gray;
        }

        button.subconfigure {
            margin-left: 5px;
            margin-right: 5px;
        }

        .tippy-content {
            color: white;
        }
    </style>
  </head>
  <body>
    <h2>AllenNLP Configuration Wizard (alpha)</h2>

    <div id="app"></div>

    <script crossorigin src="https://unpkg.com/react@16/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@16/umd/react-dom.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/immutable/3.8.2/immutable.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/6.26.0/babel.js"></script>
    <script src="https://unpkg.com/tippy.js@2.5.2/dist/tippy.all.min.js"></script>
    <script type="text/babel">
/*
The API returns objects that look like the following:

ConfigItem = {
   name : str
   annotation : List[str]
   configurable : bool
   defaultValue : str
   comment : str

Config = {
    type: str
    items: List[ConfigItem]
}

ApiResponse = {
    className: str
    config: Config
} or {
    className: str
    choices: Dict[str, str]  // name -> className
}
*/

// A configItem is optional if it has a default value
const isOptional = (configItem) => configItem.get('defaultValue') !== undefined

// Assumption is that "allennlp.*" and "torch.*" are configurable
// (at least, if they occur in a config file)
const isConfigurable = (annotation) => {
    const origin = annotation.get('origin')
    return origin.startsWith('allennlp.') || origin.startsWith('torch.')
}

// Sometimes we don't have type annotations
// (e.g. for torch classes)
// and we have to best guess how to serialize an input.
const bestGuess = x => {
    const asNumber = +x

    if (x === "true") {
        return true
    } else if (x === "false") {
        return false
    } else if (!isNaN(asNumber)) {
        return asNumber
    } else {
        // Assume string
        return x
    }
}

// Recursively convert the provided value to JSON
const jsonify = (value, annotation, configurable, optional) => {
    if (!value && optional) {
        return undefined
    } else if (configurable) {
        return configToJson(value)
    } else {
        const origin = annotation.get('origin')
        const args = annotation.get('args')

        if (origin === '?') {
            return bestGuess(value)
        }
        if (origin === 'Dict') {
            const valueAnnotation = args.get(1)
            const configurable = isConfigurable(valueAnnotation)

            const dict = {}
            let nonEmpty = false;

            (value || Immutable.List()).forEach((entry) => {
                const entryKey = entry.get("key")
                const entryValue = entry.get("value")

                if (entryKey && entryKey.length && entryValue) {
                    const valueJson = jsonify(entryValue, valueAnnotation, configurable, true)
                    if (valueJson) {
                        nonEmpty = true
                        dict[entryKey] = valueJson
                    }
                }
            })

            return (nonEmpty || !optional) ? dict : undefined
        } else if (origin === 'List' || origin === 'Sequence') {
            const [valueAnnotation] = args
            const configurable = isConfigurable(valueAnnotation)

            const list = (value || Immutable.List()).map(item => jsonify(item.get('value'), valueAnnotation, configurable, true))
                              .filter(x => x !== undefined)
                              .toArray()

            return (list.length || !optional) ? list : undefined
        } else if (origin === 'int' || origin === 'float') {
            const numeric = +value
            return value && value.length && !isNaN(numeric) ? numeric : undefined
        } else if (origin === 'bool') {
            return {'true': true, 'false': false}[value]
        } else if (origin === 'str') {
            const keep = (value && value.length) || !optional
            return keep ? (value || '') : undefined
        } else {
            console.log("unknown type " + annotation.toJS())
            return undefined
        }
    }
}

const configToJson = (config) => {

    let blob = {}

    if (config) {
        const type = config.get('type')
        if (type) {
            blob['type'] = type
        }

        config.get('items').forEach((item) => {
            const value = item.get('value')
            const annotation = item.get('annotation')
            const configurable = item.get('configurable')
            const optional = item.get('defaultValue') !== undefined

            const json = jsonify(value, annotation, configurable, optional)

            if (json !== undefined) {
                const name = item.get('name')
                blob[name] = json
            }
        })
    }

    return blob
}

// Each node in the config tree will be marked complete if either
// * it's optional, or
// * it's a leaf and has a value, or
// * all of its children are complete
//
// This means it's a bottom up concept. This is the one place
// where immutability makes our lives a little trickier.
const markComplete = (node) => {
    const optional = node.get('defaultValue') !== undefined
    const origin = node.getIn(['annotation', 'origin'])
    const value = node.get('value')

    if (node.get('configurable')) {
        if (value) {
            const items = node.getIn(['value', 'items']).map(markComplete)
            return node.set('completed', optional || items.every(item => item.get('completed')))
                    .setIn(['value', 'items'], items)
        } else {
            return node.set('completed', optional)
        }
    } else if (Immutable.List.isList(value)) {
        const items = value.map(markComplete)
        return node.set('completed', optional || items.every(item => item.get('completed')))
                   .set('value', items)
    } else {
        const completed = optional || value
        return node.set('completed', completed)
    }
}

// Top-level component
class App extends React.Component {
    constructor() {
        super()

        this.state = {
            data: Immutable.Map()
        }

        this.setData = this.setData.bind(this)
    }

    setData(fn) {
        const {data} = this.state
        const newData = markComplete(fn(data))
        return this.setState({data: newData})
    }

    componentDidMount() {
        // Fetch the top level configuration
        fetch('/api/config/')
            .then(res => res.json())
            .then(({config}) => {
                const data = Immutable.Map({configurable: true, value: Immutable.fromJS(config)})
                this.setState({data: data})
            })
    }

    render() {
        const config = this.state.data.get('value')
        return (
            <div class="wizard">
                <Config path={Immutable.List(['value'])} setData={this.setData} config={config}/>
                <JsonBox config={config}/>
            </div>
        )
    }
}

// Component that displays the JSON-rendered config file
const JsonBox = ({config}) => {
    const json = configToJson(config)

    const selectAll = () => {
        window.getSelection()
              .selectAllChildren(document.getElementById('rendered-json'))
    }

    return (
        <div id="rendered-json" onClick={selectAll}>
            {JSON.stringify(json, null, 4)}
        </div>
    )
}

// Component that renders either
// * the entire config tree, or
// * the params for a configurable class
const Config = ({path, config, setData}) => {
    if (!config) {
        return null
    }

    const type = config.get('type')
    const items = config.get('items')

    const renderedType = type ? (
        <div class="config-item">
            <span class="name">type</span>
            <span>: {type}</span>
        </div>
    ) : null

    const renderedItems = items.map((item, idx) =>
        <ConfigItem path={path.push('items', idx)} item={item} setData={setData}/>
    )

    return (
        <div class="config">
            {renderedType}
            {renderedItems}
        </div>
    )
}

// A ConfigItem will consist of a name (the key in its param dict)
// and a child that could be a text input, a list input, a dict input
// or an entire configuration. This function renders the appropriate one.
const renderChild = (path, item, setData) => {
    const annotation = item.get('annotation')

    const configurable = item.get('configurable')
    const origin = annotation.get('origin')
    const args = annotation.get('args')

    if (configurable) {
        return <Configurator path={path} item={item} setData={setData}/>
    } else if (origin === 'Dict') {
        return <Dict path={path} item={item} setData={setData}/>
    } else if (origin === 'List' || origin == 'Sequence' || (origin == 'Tuple' && args.size == 2 && args.get(2) === '...')) {
        return <List path={path} item={item} setData={setData}/>
    } else {
        return <TextInput path={path} item={item} setData={setData}/>
    }
}

class Tooltip extends React.Component {
    componentDidMount() {
        tippy('.tooltip')
    }

    render() {
        return <button class="tooltip" title={this.props.title} tabIndex="-1">?</button>
    }
}

// Represent an annotation as text
const renderAnnotation = (annotation) => {
    const origin = annotation.get('origin')
    const args = annotation.get('args')

    if (args) {
        return `${origin}[${args.map(renderAnnotation).join(', ')}]`
    } else {
        return origin
    }
}

// One of the key-value pairs that makes up a Config.
const ConfigItem = ({path, item, setData}) => {
    const name = item.get('name')
    const annotation = item.get('annotation')
    const configurable = item.get('configurable')
    const defaultValue = item.get('defaultValue')
    const comment = item.get('comment')
    const completed = item.get('completed')

    const classNames = [
        'config-item',
        defaultValue === undefined ? 'required' : 'optional',
        completed ? 'complete' : 'incomplete'
    ].join(' ')


    const tooltip = comment ? <Tooltip title={comment}/> : null
    const renderedAnnotation = <span class="annotation">({renderAnnotation(annotation)})</span>
    const renderedDefault = (defaultValue !== undefined) ? (
        <span class="default-value">{`(default: ${defaultValue})`}</span>
    ) : null

    return (
        <div className={classNames}>
            <span class="name">{name}</span>
            {tooltip}
            <span>:</span>
            {renderedAnnotation}
            {renderedDefault}
            {renderChild(path, item, setData)}
        </div>
    )
}

// Find the longest common prefix
const sharedPrefix = (s1, s2) => {
    for (let i = 0; i < s1.length; i++) {
        if (s1[i] !== s2[i]) {
            return s1.slice(0, i)
        }
    }
    // Otherwise, all of s1
    return s1
}

const commonPrefix = (strings) => {
    if (strings.size == 1) {
        // Special case, return up to last .
        const [string] = strings
        const idx = string.lastIndexOf(".")
        return string.slice(0, idx + 1)
    } else {
        return strings.reduce(sharedPrefix)
    }
}

const Configurator = ({path, item, setData}) => {
    const annotation = item.get('annotation')
    const className = annotation.get('origin')

    // User-supplied
    const choices = item.get('choices')
    const choice = item.get('choice')
    const config = item.get('value')

    const prefix = choices ? commonPrefix(choices) : null

    const getChoices = () => {
        fetch('/api/config/?class=' + className)
            .then(res => res.json())
            .then(({config, choices}) => {
                if (choices) {
                    setData(oldConfig => oldConfig.setIn(path.push('choices'), Immutable.List(choices)))
                } else {
                    setData(oldConfig => oldConfig.setIn(path.push('value'), Immutable.fromJS(config)))
                }
            })
    }

    const select = (evt) => {
        const choice = evt.target.value

        fetch('/api/config/?class=' + choice)
            .then(res => res.json())
            .then(({config}) => {
                setData(rootConfig => rootConfig.setIn(path.push('choice'), choice)
                                                .setIn(path.push('value'), Immutable.fromJS(config)))
            })
    }

    const remove = () => {
        setData(config => config.deleteIn(path.push('choices'))
                                .deleteIn(path.push('value'))
                                .deleteIn(path.push('choice')))
    }

    const configureButton = (choices || config) ? null : <button class="subconfigure" onClick={getChoices}>CONFIGURE</button>
    const choicesDropdown = choices ? (
        <span>
            <span class="prefix">{prefix}</span>
            <select value={choice} onChange={select}>
                {choice ? null : (<option value=""></option>)}
                {choices.map(subclass => <option value={subclass}>{subclass.slice(prefix.length)}</option>)}
            </select>
        </span>
    ) : null
    const deleteButton = choices ? <button onClick={remove}>x</button> : null
    const renderedValue = config ? <Config path={path.push('value')} config={config} setData={setData}/> : null

    return <span>
            {configureButton}
            <div class="choices-dropdown">
                {choicesDropdown}{deleteButton}{renderedValue}
            </div>
           </span>
}

const List = ({path, item, setData}) => {
    const [valueType] = item.get('annotation').get('args')
    const configurable = isConfigurable(valueType)
    const items = item.get('value') || Immutable.List()

    const addItem = () => {
        // We use `counter` to give every list item a unique key.
        const counter = item.get('counter') || 1

        const newItem = Immutable.Map({
            "annotation": valueType,
            "configurable": configurable,
            "itemKey": path.push(counter).join('-')
        })

        setData(config => config.setIn(path.push('value'), items.push(newItem))
                                .setIn(path.push('counter'), counter + 1))
    }

    const removeItem = idx => () => {
        setData(config => config.setIn(path.push('value'), items.delete(idx)))
    }

    const renderedItems = items.map((item, idx) => (
        <div class="list-item" key={item.get('itemKey')}>
            <button onClick={removeItem(idx)}>X</button>
            {renderChild(path.push('value', idx), item, setData)}
        </div>
    ))

    return (
        <div class="list">
            {renderedItems}
            <button onClick={addItem}>+</button>
        </div>
    )
}

const Dict = ({path, item, setData}) => {
    const [keyType, valueType] = item.get('annotation').get('args')
    const configurable = isConfigurable(valueType)

    const items = item.get('value') || Immutable.List()

    const addItem = () => {
        const counter = item.get('counter') || 1

        const newItem = Immutable.Map({
            "annotation": valueType,
            "configurable": configurable,
            "itemKey": path.push(counter).join('-')
        })

        setData(config => config.setIn(path.push('value'), items.push(newItem))
                                .setIn(path.push('counter'), counter + 1))
    }

    const removeItem = idx => () => {
        setData(config => config.setIn(path.push('value'), items.delete(idx)))
    }

    const renderedItems = items.map((item, idx) => (
        <div class="dict-item" key={item.get('itemKey')}>
            <button onClick={removeItem(idx)}>X</button>
            <TextInput path={path.push('value', idx)} fieldName="key" item={item} setData={setData}/>
            {renderChild(path.push('value', idx), item, setData)}
        </div>
    ))

    return (
        <div class="dict">
            {renderedItems}
            <button onClick={addItem}>+</button>
        </div>
    )}

const TextInput = ({path, item, setData, fieldName}) => {
    fieldName = fieldName || 'value'

    const onChange = evt => {
        setData(config => config.setIn(path.push(fieldName), evt.target.value))
    }

    return (
        <span>
            <input type="text" value={item.get(fieldName)} onChange={onChange}/>
        </span>
    )
}

ReactDOM.render(<App />, document.getElementById("app"))

    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    main(sys.argv[1:])
