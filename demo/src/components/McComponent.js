import React from 'react';
import HeatMap from './heatmap/HeatMap'
import Collapsible from 'react-collapsible'
import { API_ROOT } from '../api-config';
import { withRouter } from 'react-router-dom';
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <McInput /> Component
*******************************************************************************/

const mcExamples = [
    {
      passage: "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
      question: "How many partially reusable launch systems were developed?",
    },
    {
      passage: "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. Robotics deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing. These technologies are used to develop machines that can substitute for humans. Robots can be used in any situation and for any purpose, but today many are used in dangerous environments (including bomb detection and de-activation), manufacturing processes, or where humans cannot survive. Robots can take on any form but some are made to resemble humans in appearance. This is said to help in the acceptance of a robot in certain replicative behaviors usually performed by people. Such robots attempt to replicate walking, lifting, speech, cognition, and basically anything a human can do.",
      question: "What do robots that resemble humans attempt to do?",
    },
    {
      passage: "The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called \"the Matrix\", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer \"Neo\" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the \"dream world.\"",
      question: "Who stars in The Matrix?",
    },
    {
      passage: "Kerbal Space Program (KSP) is a space flight simulation video game developed and published by Squad for Microsoft Windows, OS X, Linux, PlayStation 4, Xbox One, with a Wii U version that was supposed to be released at a later date. The developers have stated that the gaming landscape has changed since that announcement and more details will be released soon. In the game, players direct a nascent space program, staffed and crewed by humanoid aliens known as \"Kerbals\". The game features a realistic orbital physics engine, allowing for various real-life orbital maneuvers such as Hohmann transfer orbits and bi-elliptic transfer orbits.",
      question: "What does the physics engine allow for?",
    }
];

const title = "Machine Comprehension";
const description = (
  <span>
    <span>
      Machine Comprehension (MC) answers natural language questions by selecting an answer span within an evidence text.
      The AllenNLP toolkit provides the following MC visualization, which can be used for any MC model in AllenNLP.
      This page demonstrates a reimplementation of
    </span>
    <a href = "https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Comprehen-Seo-Kembhavi/007ab5528b3bd310a80d553cccad4b78dc496b02" target="_blank" rel="noopener noreferrer">{' '} BiDAF (Seo et al, 2017)</a>
    <span>
      , or Bi-Directional Attention Flow,
      a widely used MC baseline that achieved state-of-the-art accuracies on
    </span>
    <a href = "https://rajpurkar.github.io/SQuAD-explorer/" target="_blank" rel="noopener noreferrer">{' '} the SQuAD dataset {' '}</a>
    <span>
      (Wikipedia sentences) in early 2017.
    </span>
  </span>
);


class McInput extends React.Component {
constructor(props) {
    super(props);

    // If we're showing a permalinked result,
    // we'll get passed in a passage and question.
    const { passage, question } = props;

    this.state = {
      mcPassageValue: passage || "",
      mcQuestionValue: question || ""
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handleQuestionChange = this.handleQuestionChange.bind(this);
    this.handlePassageChange = this.handlePassageChange.bind(this);
}

handleListChange(e) {
    if (e.target.value !== "") {
      this.setState({
          mcPassageValue: mcExamples[e.target.value].passage,
          mcQuestionValue: mcExamples[e.target.value].question,
      });
    }
}

handlePassageChange(e) {
    this.setState({
      mcPassageValue: e.target.value,
    });
}

handleQuestionChange(e) {
    this.setState({
    mcQuestionValue: e.target.value,
    });
}

render() {

    const { mcPassageValue, mcQuestionValue } = this.state;
    const { outputState, runMcModel } = this.props;

    const mcInputs = {
    "passageValue": mcPassageValue,
    "questionValue": mcQuestionValue
    };

    return (
        <div className="model__content">
        <ModelIntro title={title} description={description} />
            <div className="form__instructions"><span>Enter text or</span>
            <select disabled={outputState === "working"} onChange={this.handleListChange}>
                <option value="">Choose an example...</option>
                {mcExamples.map((example, index) => {
                  return (
                      <option value={index} key={index}>{example.passage.substring(0,60) + "..."}</option>
                  );
                })}
            </select>
            </div>
            <div className="form__field">
            <label htmlFor="#input--mc-passage">Passage</label>
            <textarea onChange={this.handlePassageChange} id="input--mc-passage" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius about nine times that of Earth. Although it has only one-eighth the average density of Earth, with its larger volume Saturn is just over 95 times more massive. Saturn is named after the Roman god of agriculture; its astronomical symbol represents the god&#39;s sickle.&quot;" value={mcPassageValue} disabled={outputState === "working"}></textarea>
            </div>
            <div className="form__field">
            <label htmlFor="#input--mc-question">Question</label>
            <input onChange={this.handleQuestionChange} id="input--mc-question" type="text" required="true" value={mcQuestionValue} placeholder="E.g. &quot;What does Saturn’s astronomical symbol represent?&quot;" disabled={outputState === "working"} />
            </div>
            <div className="form__field form__field--btn">
            <Button enabled={outputState !== "working"} runModel={runMcModel} inputs={mcInputs} />
            </div>
        </div>
        );
    }
}


/*******************************************************************************
  <McOutput /> Component
*******************************************************************************/

class McOutput extends React.Component {
    render() {
      const { passage, answer, attention, question_tokens, passage_tokens } = this.props;

      const start = passage.indexOf(answer);
      const head = passage.slice(0, start);
      const tail = passage.slice(start + answer.length);

      return (
        <div className="model__content">
          <div className="form__field">
            <label>Answer</label>
            <div className="model__content__summary">{ answer }</div>
          </div>

          <div className="form__field">
            <label>Passage Context</label>
            <div className="passage model__content__summary">
              <span>{head}</span>
              <span className="passage__answer">{answer}</span>
              <span>{tail}</span>
            </div>
          </div>

          <div className="form__field">
            <Collapsible trigger="Model internals (beta)">
              <Collapsible trigger="Passage to Question attention">
                <span>
                  For every passage word, the model computes an attention over the question words.
                  This heatmap shows that attention, which is normalized for every row in the matrix.
                </span>
                <div className="heatmap">
                  <HeatMap xLabels={question_tokens} yLabels={passage_tokens} data={attention} />
                </div>
              </Collapsible>
            </Collapsible>
          </div>
        </div>
      );
    }
  }


/*******************************************************************************
  <McComponent /> Component
*******************************************************************************/

class _McComponent extends React.Component {
    constructor(props) {
      super(props);

      const { requestData, responseData } = props;

      this.state = {
        outputState: responseData ? "received" : "empty", // valid values: "working", "empty", "received", "error"
        requestData: requestData,
        responseData: responseData
      };

      this.runMcModel = this.runMcModel.bind(this);
    }

    runMcModel(event, inputs) {
      this.setState({outputState: "working"});

      var payload = {
        passage: inputs.passageValue,
        question: inputs.questionValue,
      };
      fetch(`${API_ROOT}/predict/machine-comprehension`, {
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
        const newPath = slug ? '/machine-comprehension/' + slug : '/machine-comprehension';

        // We'll pass the request and response data along as part of the location object
        // so that the `Demo` component can use them to re-render.
        const location = {
          pathname: newPath,
          state: { requestData: payload, responseData: json }
        }
        this.props.history.push(location);
      }).catch((error) => {
        this.setState({outputState: "error"});
        console.error(error);
      });
    }

    render() {
      const { requestData, responseData } = this.props;

      const passage = requestData && requestData.passage;
      const question = requestData && requestData.question;
      const answer = responseData && responseData.best_span_str;
      const attention = responseData && responseData.passage_question_attention;
      const question_tokens = responseData && responseData.question_tokens;
      const passage_tokens = responseData && responseData.passage_tokens;

      return (
        <div className="pane model">
          <PaneLeft>
            <McInput runMcModel={this.runMcModel}
                     outputState={this.state.outputState}
                     passage={passage}
                     question={question}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <McOutput passage={passage}
                      answer={answer}
                      attention={attention}
                      question_tokens={question_tokens}
                      passage_tokens={passage_tokens}/>
          </PaneRight>
        </div>
      );

    }
}

const McComponent = withRouter(_McComponent);

export default McComponent;
