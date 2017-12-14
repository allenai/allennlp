import React from 'react';
import { API_ROOT } from '../api-config';
import { withRouter } from 'react-router-dom';
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <TeInput /> Component
*******************************************************************************/

const teExamples = [
    {
      premise: "If you help the needy, God will reward you.",
      hypothesis: "Giving money to the poor has good consequences.",
    },
    {
      premise: "Two women are wandering along the shore drinking iced tea.",
      hypothesis: "Two women are sitting on a blanket near some rocks talking about politics.",
    },
    {
      premise: "An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
      hypothesis: "The spacecraft has the ability to travel between planets.",
    },
    {
      premise: "A large, gray elephant walked beside a herd of zebras.",
      hypothesis: "The elephant was lost.",
    },
    {
      premise: "A handmade djembe was on display at the Smithsonian.",
      hypothesis: "Visitors could see the djembe.",
    },
  ];

  const title = "Textual Entailment";
  const description = (
    <span>
      <span>
        Textual Entailment (TE) takes a pair of sentences and predicts whether the facts in the first
        necessarily imply the facts in the second one.  The AllenNLP toolkit provides the following TE visualization,
        which can be run for any TE model you develop.
        This page demonstrates a reimplementation of
      </span>
      <a href = "https://www.semanticscholar.org/paper/A-Decomposable-Attention-Model-for-Natural-Languag-Parikh-T%C3%A4ckstr%C3%B6m/07a9478e87a8304fc3267fa16e83e9f3bbd98b27" target="_blanke" rel="noopener noreferrer">{' '} the decomposable attention model (Parikh et al, 2017) {' '}</a>
      <span>
        , which was state of the art for
      </span>
      <a href = "https://nlp.stanford.edu/projects/snli/" target="_blank" rel="noopener noreferrer">{' '} the SNLI benchmark {' '}</a>
      <span>
        (short sentences about visual scenes) in 2016.
      </span>
    </span>
  );

  class TeInput extends React.Component {
    constructor(props) {
      super(props);

    // If we're showing a permalinked result,
    // we'll get passed in a premise and hypothesis.
    const { premise, hypothesis } = props;

      this.state = {
        tePremiseValue: premise || "",
        teHypothesisValue: hypothesis || "",
      };
      this.handleListChange = this.handleListChange.bind(this);
      this.handlePremiseChange = this.handlePremiseChange.bind(this);
      this.handleHypothesisChange = this.handleHypothesisChange.bind(this);
    }

    handleListChange(e) {
      if (e.target.value !== "") {
        this.setState({
          tePremiseValue: teExamples[e.target.value].premise,
          teHypothesisValue: teExamples[e.target.value].hypothesis,
        });
      }
    }

    handlePremiseChange(e) {
      this.setState({
        tePremiseValue: e.target.value,
      });
    }

    handleHypothesisChange(e) {
      this.setState({
        teHypothesisValue: e.target.value,
      });
    }

    render() {
      const { tePremiseValue, teHypothesisValue } = this.state;
      const { outputState, runTeModel } = this.props;

      const teInputs = {
        "premiseValue": tePremiseValue,
        "hypothesisValue": teHypothesisValue
      };

      return (
        <div className="model__content">
        <ModelIntro title={title} description={description} />
          <div className="form__instructions"><span>Enter text or</span>
            <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {teExamples.map((example, index) => {
                return (
                  <option value={index} key={index}>{example.premise}</option>
                );
              })}
            </select>
          </div>
          <div className="form__field">
            <label htmlFor="input--te-premise">Premise</label>
            <input onChange={this.handlePremiseChange} id="input--te-premise" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;A black dog chased a cat throught the park.&quot;" value={tePremiseValue} />
          </div>
          <div className="form__field">
            <label htmlFor="input--te-hypothesis">Hypothesis</label>
            <input onChange={this.handleHypothesisChange} id="input--te-hypothesis" type="text" required="true" value={teHypothesisValue} placeholder="E.g. &quot;The cat is black.&quot;" />
          </div>
          <div className="form__field form__field--btn">
            <Button enabled={outputState !== "working"} runModel={runTeModel} inputs={teInputs} />
          </div>
        </div>
      );
    }
  }


/*******************************************************************************
  <TeGraph /> Component
*******************************************************************************/

class TeGraph extends React.Component {
    render() {
      const { x, y } = this.props

      const width = 224;
      const height = 194;

      const absoluteX = Math.round(x * width);
      const absoluteY = Math.round((1.0 - y) * height);

      const plotCoords = {
        left: `${absoluteX}px`,
        top: `${absoluteY}px`,
      };

      return (
        <div className="te-graph-labels">
          <div className="te-graph">
            <div className="te-graph__point" style={plotCoords}></div>
          </div>
        </div>
      );
    }
}

  /*******************************************************************************
  <TeOutput /> Component
*******************************************************************************/

class TeOutput extends React.Component {
    render() {
      const { labelProbs } = this.props;
      const [entailment, contradiction, neutral] = labelProbs;

      let judgment; // Valid values: "e", "c", "n"
      let degree; // Valid values: "somewhat", "very"

      if (entailment > contradiction && entailment > neutral) {
        judgment = "e"
      }
      else if (contradiction > entailment && contradiction > neutral) {
        judgment = "c"
      }
      else if (neutral > entailment && neutral > contradiction) {
        judgment = "n"
      }

      const veryConfident = 0.75;
      const somewhatConfident = 0.50;
      const summaryText = () => {
        if (entailment >= veryConfident || contradiction >= veryConfident || neutral >= veryConfident) {
          let judgmentStr;
          switch(judgment) {
            case "c":
              judgmentStr = (<span>the premise <strong>contradicts</strong> the hypothesis</span>);
              break;
            case "e":
              judgmentStr = (<span>the premise <strong>entails</strong> the hypothesis</span>);
              break;
            case "n":
              judgmentStr = (<span>there is <strong>no correlation</strong> between the premise and hypothesis</span>);
              break;
            default:
              throw new Error("Unhandled case for judgement confidence.")
          }
          return (
            <div className="model__content__summary">It is <strong>{degree} likely</strong> that {judgmentStr}.</div>
          );
        }
        else if (entailment >= somewhatConfident || contradiction >= somewhatConfident || neutral >= somewhatConfident) {
          let judgmentStr;
          switch(judgment) {
            case "c":
              judgmentStr = (<span>the premise <strong>contradicts</strong> the hypothesis</span>);
              break;
            case "e":
              judgmentStr = (<span>the premise <strong>entails</strong> the hypothesis</span>);
              break;
            case "n":
              judgmentStr = (<span>there is <strong>no correlation</strong> between the premise and hypothesis</span>);
              break;
            default:
              throw new Error("Unhandled case for judgement correlation.")
          }
          return (
            <div className="model__content__summary">It is <strong>somewhat likely</strong> that {judgmentStr}.</div>
          );
        }
        else {
          return (
            <div className="model__content__summary">The model is not confident in its judgment.</div>
          );
        }
      }

      function formatProb(n) {
        return parseFloat((n * 100).toFixed(1)) + "%";
      }

      // https://en.wikipedia.org/wiki/Ternary_plot#Plotting_a_ternary_plot
      const a = contradiction;
      const b = neutral;
      const c = entailment;
      const x = 0.5 * (2 * b + c) / (a + b + c)
      const y = (c / (a + b + c))

      return (
        <div className="model__content">
          <div className="form__field">
            <label>Summary</label>
            {summaryText()}
          </div>
          <div className="te-output">
            <TeGraph x={x} y={y}/>
            <div className="te-table">
              <table>
                <thead>
                  <tr>
                    <th>Judgement</th>
                    <th>Probability</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Entailment</td>
                    <td>{formatProb(entailment)}</td>
                  </tr>
                  <tr>
                    <td>Contradiction</td>
                    <td>{formatProb(contradiction)}</td>
                  </tr>
                  <tr>
                    <td>Neutral</td>
                    <td>{formatProb(neutral)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      );
    }
}

/*******************************************************************************
  <TeComponent /> Component
*******************************************************************************/

class _TeComponent extends React.Component {
    constructor(props) {
      super(props);

      const { requestData, responseData } = props;

      this.state = {
        outputState: responseData ? "received" : "empty", // valid values: "working", "empty", "received", "error"
        requestData: requestData,
        responseData: responseData
      };

      this.runTeModel = this.runTeModel.bind(this);
    }

    runTeModel(event, inputs) {
      this.setState({outputState: "working"});

      var payload = {
        premise: inputs.premiseValue,
        hypothesis: inputs.hypothesisValue,
      };
      fetch(`${API_ROOT}/predict/textual-entailment`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      }).then((response) => {
        return response.json();
      }).then((json) => {
        // If the response contains a `slug` for a permalink, we want to redirect
        // to the corresponding path using `history.push`.
        const { slug } = json;
        const newPath = slug ? '/textual-entailment/' + slug : '/textual-entailment';

        // We'll pass the request and response data along as part of the location object
        // so that the `Demo` component can use them to re-render.
        const location = {
          pathname: newPath,
          state: {requestData: payload, responseData: json}
        }
        this.props.history.push(location);
      }).catch((error) => {
        this.setState({outputState: "error"});
        console.error(error);
      });
    }

    render() {
      const { requestData, responseData } = this.props;

      // Get inputs and outputs, which may be null.
      const premise = requestData && requestData.premise;
      const hypothesis = requestData && requestData.hypothesis;
      const labelProbs = responseData && responseData.label_probs;

      return (
        <div className="pane model">
          <PaneLeft>
            <TeInput runTeModel={this.runTeModel}
                     outputState={this.state.outputState}
                     premise={premise}
                     hypothesis={hypothesis}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <TeOutput labelProbs={labelProbs}/>
          </PaneRight>
        </div>
      );
    }
}

const TeComponent = withRouter(_TeComponent);

export default TeComponent;
