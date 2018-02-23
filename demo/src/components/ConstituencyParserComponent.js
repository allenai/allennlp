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
      A constituency parse tree breaks a text into sub-phrases, or constituents. Non-terminals in the tree are types of phrases, the terminals are the words in the sentence.
      This demo is an implementation of a minimal neural model for constituency parsing based on an independent scoring of labels and spans from 
    </span>
    <a href="https://www.semanticscholar.org/paper/A-Minimal-Span-Based-Neural-Constituency-Parser-Stern-Andreas/593e4e749bd2dbcaf8dc25298d830b41d435e435" target="_blank" rel="noopener noreferrer">{' '} a Minimal Span Based Constituency Parser (Stern et al, 2017)</a>
    <span>
      . This model achieved state of the art single model performance on the Penn Tree Bank in 2017.
    </span> 
  </span>
);

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
          <input onChange={this.handleSentenceChange} value={constituencyParserSentenceValue} id="input--parser-sentence" ref="constituencyParserSentence" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;John likes and Bill hates ice cream.&quot;" />
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


class _ConstituencyParserComponent extends React.Component {
  constructor(props) {
    super(props);

    const { requestData, responseData } = props;

    this.state = {
      requestData: requestData,
      responseData: responseData,
      // valid values: "working", "empty", "received", "error",
      outputState: responseData ? "received" : "empty",
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
    const sentence = requestData && requestData.sentence;

    return (
      <div className="pane model">
        <PaneLeft>
          <ConstituencyParserInput runConstituencyParserModel={this.runConstituencyParserModel}
            outputState={this.state.outputState}
            sentence={sentence} />
        </PaneLeft>
        <PaneRight outputState={this.state.outputState}>
          <HierplaneVisualization tree={responseData ? responseData.hierplane_tree : null} />
        </PaneRight>
      </div>
    );
  }
}

const ConstituencyParserComponent = withRouter(_ConstituencyParserComponent)

export default ConstituencyParserComponent;
