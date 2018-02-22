import React from 'react';
import { API_ROOT } from '../api-config';
import { withRouter } from 'react-router-dom';
import { PaneLeft, PaneRight } from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'
import { Tree } from 'hierplane';

/*******************************************************************************
  <ConstituencyParserInput /> Component
*******************************************************************************/

const constituencyParserSentences = [
  "The keys, which were needed to access the building, were locked in the car.",
  "However, voters decided that if the stadium was such a good idea someone would build it himself, and rejected it 59% to 41%.",
  "Did Uriah honestly think he could beat the game in under three hours?",
  "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!",
  "More than a few CEOs say the red-carpet treatment tempts them to return to a heartland city for future meetings.",
];

const title = "Constituency Parsing";
const description = (
  <span>
    <span>
        Constituency Parsing yo
    </span>
    <a href="https://www.semanticscholar.org/paper/Deep-Semantic-Role-Labeling-What-Works-and-What-s-He-Lee/a3ccff7ad63c2805078b34b8514fa9eab80d38e9" target="_blank" rel="noopener noreferrer">{' '} a deep BiLSTM model (He et al, 2017)</a>
  </span>
);

function getStrIndex(words, wordIdx) {
  if (wordIdx < 0) throw new Error(`Invalid word index: ${wordIdx}`);
  return words.slice(0, wordIdx).join(' ').length;
}

class ConstituencyParserInput extends React.Component {
  constructor(props) {
    super(props);

    // If we're showing a permalinked result, we'll get passed in a sentence.
    const { sentence } = props;

    this.state = {
      sentenceValue: sentence || "",
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handleSentenceChange = this.handleSentenceChange.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      this.setState({
        constituencyParserSentenceValue: constituencyParserSentences[e.target.value],
      });
    }
  }

  handleSentenceChange(e) {
    this.setState({
      constituencyParserSentenceValue: e.target.value,
    });
  }

  render() {
    const { constituencyParserSentenceValue } = this.state;
    const { outputState, runConstituencyParserModel } = this.props;

    const constituencyParserInputs = {
      "sentenceValue": constituencyParserSentenceValue,
    };

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
            <option>Choose an example...</option>
            {constituencyParserSentences.map((sentence, index) => {
              return (
                <option value={index} key={index}>{sentence}</option>
              );
            })}
          </select>
        </div>
        <div className="form__field">
          <label htmlFor="#input--srl-sentence">Sentence</label>
          <input onChange={this.handleSentenceChange} value={constituencyParserSentenceValue} id="input--srl-sentence" ref="constituencyParserSentence" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;John likes and Bill hates ice cream.&quot;" />
        </div>
        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} outputState={outputState} runModel={runConstituencyParserModel} inputs={constituencyParserInputs} />
        </div>
      </div>
    );
  }
}

class HierplaneVisualization extends React.Component {
  render() {
    if (this.props.tree) {
      return (
        <div className="hierplane__visualization">
          <div className="hierplane__visualization-verbs">
            <a className="hierplane__visualization-verbs__prev">
              <svg width="12" height="12">
                <use xlinkHref="#icon__disclosure"></use>
              </svg>
            </a>
          </div>
          <Tree tree={this.props.tree} theme="light" />
        </div>
      )
    } else {
      return null;
    }
  }
}

/*******************************************************************************
  <ConstituencyParserComponent /> Component
*******************************************************************************/

const VisualizationType = {
  TREE: 'Tree',
  TEXT: 'Text'
};
Object.freeze(VisualizationType);

class _ConstituencyParserComponent extends React.Component {
  constructor(props) {
    super(props);

    const { requestData, responseData } = props;

    this.state = {
      requestData: requestData,
      responseData: responseData,
      // valid values: "working", "empty", "received", "error",
      outputState: responseData ? "received" : "empty",
      visualizationType: VisualizationType.TREE
    };

    this.runConstituencyParserModel = this.runConstituencyParserModel.bind(this);
  }

  runConstituencyParserModel(event, inputs) {
    this.setState({outputState: "working"});

    var payload = {sentence: inputs.sentenceValue};

    fetch(`${API_ROOT}/predict/constituency-parsing`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    }).then(function (response) {
      return response.json();
    }).then((json) => {
      // If the response contains a `slug` for a permalink, we want to redirect
      // to the corresponding path using `history.push`.
      const { slug } = json;
      const newPath = slug ? '/constituency-parsing/' + slug : '/constituency-parsing';

      // We'll pass the request and response data along as part of the location object
      // so that the `Demo` component can use them to re-render.
      const location = {
        pathname: newPath,
        state: { requestData: payload, responseData: json }
      }
      this.props.history.push(location);
    }).catch((error) => {
      this.setState({ outputState: "error" });
      console.error(error);
    });
  }

  render() {
    const { requestData, responseData } = this.props;
    const { visualizationType } = this.state;

    const sentence = requestData && requestData.sentence;
    console.log(responseData);
    let viz = null;
    switch(visualizationType) {
      case VisualizationType.TEXT:
        viz = <div><p>{responseData.trees}</p></div>;
        break;
      case VisualizationType.TREE:
      default:
        viz = <HierplaneVisualization tree={responseData ? responseData.trees : null} />
        break;
    }

    return (
      <div className="pane model">
        <PaneLeft>
          <ConstituencyParserInput runConstituencyParserModel={this.runConstituencyParserModel}
            outputState={this.state.outputState}
            sentence={sentence} />
        </PaneLeft>
        <PaneRight outputState={this.state.outputState}>
          <ul className="srl__vizualization-types">
            {Object.keys(VisualizationType).map(tpe => {
              const vizType = VisualizationType[tpe];
              const className = (
                visualizationType === vizType
                  ? 'srl__vizualization-types__active-type'
                  : null
              );
              return (
                <li key={vizType} className={className}>
                  <a onClick={() => this.setState({ visualizationType: vizType })}>
                    {vizType}
                  </a>
                </li>
              );
            })}
          </ul>
          {viz}
        </PaneRight>
      </div>
    );
  }
}

const ConstituencyParserComponent = withRouter(_ConstituencyParserComponent)

export default ConstituencyParserComponent;
