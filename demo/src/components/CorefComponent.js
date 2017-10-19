import React from 'react';
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <CorefInput /> Component
*******************************************************************************/

const corefExamples = [
    {
      passage: "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
    },
    {
      passage: "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. Robotics deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing. These technologies are used to develop machines that can substitute for humans. Robots can be used in any situation and for any purpose, but today many are used in dangerous environments (including bomb detection and de-activation), manufacturing processes, or where humans cannot survive. Robots can take on any form but some are made to resemble humans in appearance. This is said to help in the acceptance of a robot in certain replicative behaviors usually performed by people. Such robots attempt to replicate walking, lifting, speech, cognition, and basically anything a human can do.",
    },
    {
      passage: "The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called \"the Matrix\", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer \"Neo\" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the \"dream world.\"",
    },
    {
      passage: "Kerbal Space Program (KSP) is a space flight simulation video game developed and published by Squad for Microsoft Windows, OS X, Linux, PlayStation 4, Xbox One, with a Wii U version that was supposed to be released at a later date. The developers have stated that the gaming landscape has changed since that announcement and more details will be released soon. In the game, players direct a nascent space program, staffed and crewed by humanoid aliens known as \"Kerbals\". The game features a realistic orbital physics engine, allowing for various real-life orbital maneuvers such as Hohmann transfer orbits and bi-elliptic transfer orbits.",
    }
];

class CorefInput extends React.Component {
constructor() {
    super();
    this.state = {
    corefPassageValue: "",
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handlePassageChange = this.handlePassageChange.bind(this);
}

handleListChange(e) {
    if (e.target.value !== "") {
    this.setState({
        corefPassageValue: corefExamples[e.target.value].passage,
    });
    }
}

handlePassageChange(e) {
    this.setState({
    corefPassageValue: e.target.value,
    });
}

render() {

    const { corefPassageValue } = this.state;
    const { outputState, runCorefModel } = this.props;

    const corefInputs = {
    "passageValue": corefPassageValue,
    };

    const title = "Co-reference Resolution";
    const description = (
      <div>
        <span>
        Coreference resolution is the task of finding all expressions that refer to the same entity
        in a text. It is an important step for a lot of higher level NLP tasks that involve natural
        language understanding such as document summarization, question answering, and
        information extraction.
        </span>
        <a href = "https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83" target="_blank" rel="noopener noreferrer">{' '} End-to-end Neural Coreference Resolution ( Lee et al, 2017)</a>
        <span>
        a widely used MC baseline that achieved state-of-the-art accuracies on
        </span>
        <a href = "http://cemantix.org/data/ontonotes.html" target="_blank" rel="noopener noreferrer">{' '} the Ontonotes 5.0 dataset {' '}</a>
        <span>
        in early 2017.
        </span>
      </div>
    );

    return (
        <div className="model__content">
        <ModelIntro title={title} description={description} />
            <div className="form__instructions"><span>Enter text or</span>
            <select disabled={outputState === "working"} onChange={this.handleListChange}>
                <option value="">Choose an example...</option>
                {corefExamples.map((example, index) => {
                return (
                    <option value={index} key={index}>{example.passage.substring(0,60) + "..."}</option>
                );
                })}
            </select>
            </div>
            <div className="form__field">
            <label htmlFor="#input--mc-passage">Passage</label>
            <textarea onChange={this.handlePassageChange} id="input--mc-passage" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius about nine times that of Earth. Although it has only one-eighth the average density of Earth, with its larger volume Saturn is just over 95 times more massive. Saturn is named after the Roman god of agriculture; its astronomical symbol represents the god&#39;s sickle.&quot;" value={corefPassageValue} disabled={outputState === "working"}></textarea>
            </div>
            <div className="form__field form__field--btn">
            <Button outputState={outputState} runModel={runCorefModel} inputs={corefInputs} />
            </div>
        </div>
        );
    }
}

/*******************************************************************************
  <CorefOutput /> Component
*******************************************************************************/

class CorefOutput extends React.Component {
    render() {
      const { passage, answer } = this.props;
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
        </div>
      );
    }
  }


/*******************************************************************************
  <CorefComponent /> Component
*******************************************************************************/

class CorefComponent extends React.Component {
    constructor() {
      super();

      this.state = {
        outputState: "empty", // valid values: "working", "empty", "received", "error"
        passage: "",
        answer: "",
      };

      this.runCorefModel = this.runCorefModel.bind(this);
    }

    runCorefModel(event, inputs) {
      this.setState({
        outputState: "working",
      });

      var payload = {
        passage: inputs.passageValue,
        question: inputs.questionValue,
      };
      fetch('/predict/coreference-resolution', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      }).then((response) => {
        return response.json();
      }).then((json) => {
        this.setState({answer: json["best_span_str"]});
        this.setState({passage: inputs.passageValue});
        this.setState({outputState: "received"});
      }).catch((error) => {
        this.setState({outputState: "error"});
        throw error; // todo(michaels): is this right?
      });
    }

    render() {
      return (
        <div className="pane model">
          <PaneLeft>
            <CorefInput runCorefModel={this.runCorefModel} outputState={this.state.outputState}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <CorefOutput answer={this.state.answer} passage={this.state.passage} />
          </PaneRight>
        </div>
      );
    }
}

export default CorefComponent;