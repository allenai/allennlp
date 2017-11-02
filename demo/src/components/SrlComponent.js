import React from 'react';
import { withRouter } from 'react-router-dom';
import { PaneLeft, PaneRight } from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <SrlInput /> Component
*******************************************************************************/

const srlSentences = [
  "The keys, which were needed to access the building, were locked in the car.",
  "However, voters decided that if the stadium was such a good idea someone would build it himself, and rejected it 59% to 41%.",
  "Did Uriah honestly think he could beat the game in under three hours?",
  "If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!",
  "More than a few CEOs say the red-carpet treatment tempts them to return to a heartland city for future meetings.",
];

const title = "Semantic Role Labeling";
const description = (
  <div>
    <span>
      Semantic Role Labeling (SRL) recovers the latent predicate argument structure of a sentence,
      providing representations that answer basic questions about sentence meaning, including “who” did “what” to “whom,” etc.
      The AllenNLP toolkit provides the following SRL visualization, which can be used for any SRL model in AllenNLP.
      This page demonstrates a reimplementation of
    </span>
    <a href="https://www.semanticscholar.org/paper/Deep-Semantic-Role-Labeling-What-Works-and-What-s-He-Lee/a3ccff7ad63c2805078b34b8514fa9eab80d38e9" target="_blank" rel="noopener noreferrer">{' '} a deep BiLSTM model (He et al, 2017)</a>
    <span>
      , which is currently state of the art for PropBank SRL (Newswire sentences).
    </span>
  </div>
);

class SrlInput extends React.Component {
  constructor(props) {
    super(props);

    // If we're showing a permalinked result, we'll get passed in a sentence.
    const { sentence } = props;

    this.state = {
      srlSentenceValue: sentence || "",
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handleSentenceChange = this.handleSentenceChange.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      this.setState({
        srlSentenceValue: srlSentences[e.target.value],
      });
    }
  }

  handleSentenceChange(e) {
    this.setState({
      srlSentenceValue: e.target.value,
    });
  }

  render() {
    const { srlSentenceValue } = this.state;
    const { outputState, runSrlModel } = this.props;

    const srlInputs = {
      "sentenceValue": srlSentenceValue,
    };

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
            <option>Choose an example...</option>
            {srlSentences.map((sentence, index) => {
              const selected = sentence === srlSentenceValue;
              return (
                <option value={index} key={index} selected={selected}>{sentence}</option>
              );
            })}
          </select>
        </div>
        <div className="form__field">
          <label htmlFor="#input--srl-sentence">Sentence</label>
          <input onChange={this.handleSentenceChange} value={srlSentenceValue} id="input--srl-sentence" ref="srlSentence" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;John likes and Bill hates ice cream.&quot;" />
        </div>
        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} outputState={outputState} runModel={runSrlModel} inputs={srlInputs} />
        </div>
      </div>
    );
  }
}


/*******************************************************************************
  <SrlOutput /> Component
*******************************************************************************/

// Render the SRL tag for a single word as a table cell
class SrlTagCell extends React.Component {

  render() {
    const { tag, colorClass } = this.props;

    // Don't show "O" tags, and slice off all the "B-" and "I-" prefixes.
    const tagText = tag === "O" ? "" : tag.slice(2);

    return (
      <td className={colorClass + ' srl-tag srl-tag-' + tag.toLowerCase()}>
        {tagText}
      </td>
    )
  }
}

// Render a SRL-tagged word as a table cell
class SrlWordCell extends React.Component {
  render() {
    const { word, colorClass } = this.props;

    return (<td className={colorClass + ' srl-word'}>{word}</td>)
  }
}

class SrlFrame extends React.Component {
  render() {
    const { verb, words } = this.props;
    const tags = verb["tags"];


    // Skip frames that have only one tag; these are typically helper verbs.
    // In an ideal world we'd filter these out on the backend, but the POS
    // tagger we're using right now doesn't seem up to the task.
    const numTags = tags.filter(tag => tag !== "O").length
    if (numTags <= 1) {
      return (<div />)
    }

    // Create an array indicating what color to highlight each tag cell.
    // For "O" tags this should be -1, indicating no color.
    // Otherwise it should toggle between 0 and 1 every time a "B-" tag occurs.
    var colorClasses = [];
    var currentColor = 1;

    tags.forEach(function (tag, i) {
      if (tag === "O") {
        // "O" tag, so append "" for "no color"
        colorClasses.push("");
      } else if (tag[0] === "B") {
        // "B-" tag, so toggle the current color and then append
        currentColor = (currentColor + 1) % 2;
        colorClasses.push("color" + currentColor);
      } else /* (tag[0] == "I") */ {
        // "I-" tag, so append the current color
        colorClasses.push("color" + currentColor);
      }
    })

    return (
      <div>
        <label>{verb.verb}</label>
        <table className="srl-table">
          <tbody>
            <tr>
              {tags.map((tag, i) => <SrlTagCell tag={tag} key={i} colorClass={colorClasses[i]} />)}
            </tr>
            <tr>
              {words.map((word, i) => <SrlWordCell word={word} key={i} colorClass={colorClasses[i]} />)}
            </tr>
          </tbody>
        </table>
      </div>
    )
  }
}

class SrlOutput extends React.Component {
  render() {
    const { words, verbs } = this.props;

    return (
      <div className="model__content model__content--srl-output">
        <div className="form__field">
          {verbs.map((verb, i) => (<SrlFrame verb={verb} words={words} key={i} />))}
        </div>
      </div>
    );
  }
}

/*******************************************************************************
  <SrlComponent /> Component
*******************************************************************************/

class _SrlComponent extends React.Component {
  constructor(props) {
    super(props);

    const { requestData, responseData } = props;

    this.state = {
      outputState: this.props.responseData ? "received" : "empty", // valid values: "working", "empty", "received", "error"
      requestData: requestData,
      responseData: responseData
    };

    this.runSrlModel = this.runSrlModel.bind(this);
  }

  runSrlModel(event, inputs) {
    this.setState({outputState: "working"});

    var payload = {sentence: inputs.sentenceValue};

    fetch('http://localhost:8000/predict/semantic-role-labeling', {
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
      const newPath = slug ? '/semantic-role-labeling/' + slug : '/semantic-role-labeling';

      // We'll pass the request and response data along as part of the location object
      // so that the `Demo` component can use them to re-render.
      const location = {
        pathname: newPath,
        state: { requestData: payload, responseData: json }
      }
      this.props.history.push(location);
    }).catch((error) => {
      this.setState({ outputState: "error" });
      throw error; // todo(michaels): is this right?
    });
  }

  render() {
    const { requestData, responseData } = this.props;

    const sentence = requestData && requestData.sentence;
    const words = responseData && responseData.words;
    const verbs = responseData && responseData.verbs;

    return (
      <div className="pane model">
        <PaneLeft>
          <SrlInput runSrlModel={this.runSrlModel}
            outputState={this.state.outputState}
            sentence={sentence} />
        </PaneLeft>
        <PaneRight outputState={this.state.outputState}>
          <SrlOutput words={words} verbs={verbs} />
        </PaneRight>
      </div>
    );
  }
}

const SrlComponent = withRouter(_SrlComponent)

export default SrlComponent;
