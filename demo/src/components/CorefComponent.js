import React from 'react';
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <CorefInput /> Component
*******************************************************************************/

const corefExamples = [
    {
      document: "is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. Robotics deals with the design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing. These technologies are used to develop machines that can substitute for humans. Robots can be used in any situation and for any purpose, but today many are used in dangerous environments (including bomb detection and de-activation), manufacturing processes, or where humans cannot survive. Robots can take on any form but some are made to resemble humans in appearance. This is said to help in the acceptance of a robot in certain replicative behaviors usually performed by people. Such robots attempt to replicate walking, lifting, speech, cognition, and basically anything a human can do.",
    },
    {
      document: "The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called \"the Matrix\", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer \"Neo\" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the \"dream world.\"",
    },
    {
      document: "Kerbal Space Program (KSP) is a space flight simulation video game developed and published by Squad for Microsoft Windows, OS X, Linux, PlayStation 4, Xbox One, with a Wii U version that was supposed to be released at a later date. The developers have stated that the gaming landscape has changed since that announcement and more details will be released soon. In the game, players direct a nascent space program, staffed and crewed by humanoid aliens known as \"Kerbals\". The game features a realistic orbital physics engine, allowing for various real-life orbital maneuvers such as Hohmann transfer orbits and bi-elliptic transfer orbits.",
    }
];

class CorefInput extends React.Component {
    constructor() {
        super();
        this.state = {
        corefDocumentValue: "",
        };
        this.handleListChange = this.handleListChange.bind(this);
        this.handleDocumentChange = this.handleDocumentChange.bind(this);
    }

    handleListChange(e) {
        if (e.target.value !== "") {
        this.setState({
            corefDocumentValue: corefExamples[e.target.value].document,
        });
        }
    }

    handleDocumentChange(e) {
        this.setState({
        corefDocumentValue: e.target.value,
        });
    }

    render() {
        const { corefDocumentValue } = this.state;
        const { outputState, runCorefModel } = this.props;

        const corefInputs = {
        "documentValue": corefDocumentValue,
        };

        const title = "Co-reference Resolution";
        const description = (
          <div>
            <span>
            Coreference resolution is the task of finding all expressions that refer to the same entity
            in a text. It is an important step for a lot of higher level NLP tasks that involve natural
            language understanding such as document summarization, question answering, and information extraction.
            </span>
            <a href = "https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83" target="_blank" rel="noopener noreferrer">{' '} End-to-end Neural Coreference Resolution ( Lee et al, 2017)</a>
            <span>
            a neural model which considers all possible spans in the document as potential mentions and
            learns distributions over possible anteceedents for each span, using aggressive, learnt
            pruning strategies to retain computational efficiency. It achieved state-of-the-art accuracies on
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
                        <option value={index} key={index}>{example.document.substring(0,60) + ".. ."}</option>
                    );
                    })}
                </select>
                </div>
                <div className="form__field">
                <label htmlFor="#input--mc-passage">Document</label>
                <textarea onChange={this.handleDocumentChange} id="input--mc-passage" type="text"
                required="true" autoFocus="true" placeholder="E.g. &quot;Saturn is the sixth planet from the Sun and the second-largest in the Solar System, after Jupiter. It is a gas giant with an average radius about nine times that of Earth. Although it has only one-eighth the average density of Earth, with its larger volume Saturn is just over 95 times more massive. Saturn is named after the Roman god of agriculture; its astronomical symbol represents the god&#39;s sickle.&quot;" value={corefDocumentValue} disabled={outputState === "working"}></textarea>
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
    constructor() {
      super();
      this.state = {
        selectedCluster: -1,
      };
      this.onClusterMouseover = this.onClusterMouseover.bind(this);
    }

    onClusterMouseover(index) {
      this.setState( { selectedCluster: index });
    }

    render() {
      const { document, clusters } = this.props;
      var clusteredDocument = document.map((word, wordIndex) => {
        var membershipClusters = [];
        clusters.forEach((cluster, clusterIndex) => {
          cluster.forEach((span) => {
            if (wordIndex >= span[0] && wordIndex <= span[1]) {
              membershipClusters.push(clusterIndex);
            }
          });
        });
        return { word : word, clusters : membershipClusters }
      });

      var wordStyle = (clusteredWord) => {
        var clusters = clusteredWord['clusters'];

        if (clusters.includes(this.state.selectedCluster)) {
          return "passage__answer";
        }
        else {
          return "unselected";
        }
      }

      return (
        <div className="model__content">
          <div className="form__field">
            <label>Clusters</label>
            <div className="model__content__summary">
            <ul>
              {clusters.map((cluster, index) =>
               <li>
                {cluster.map((span, wordIndex) =>
                <a href="#" key={ wordIndex } onMouseEnter={ () => this.onClusterMouseover(index) }> {document.slice(span[0], span[1] + 1).join(" ")}, </a>
                )}
               </li>
            )}
            </ul>
            </div>
          </div>

          <div className="form__field">
            <label>Document</label>
            <div className="passage model__content__summary">
            {clusteredDocument.map((clusteredWord, index) =>
              <span className={ wordStyle(clusteredWord) }> {clusteredWord['word']}</span>
            )}
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
        document: "",
        clusters: {},
      };

      this.runCorefModel = this.runCorefModel.bind(this);
    }

    runCorefModel(event, inputs) {
      this.setState({
        outputState: "working",
      });

      var payload = {
        document: inputs.documentValue,
      };

      fetch('http://localhost:8000/predict/coreference-resolution', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      }).then((response) => {
        return response.json();
      }).then((json) => {
        this.setState({clusters: json["clusters"]});
        this.setState({document: json["document"]});
        this.setState({outputState: "received"});
      }).catch((error) => {
        this.setState({outputState: "error"});
        console.error(error);
      });
    }

    render() {
      return (
        <div className="pane model">
          <PaneLeft>
            <CorefInput runCorefModel={this.runCorefModel} outputState={this.state.outputState}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <CorefOutput clusters={this.state.clusters} document={this.state.document} />
          </PaneRight>
        </div>
      );
    }
}

export default CorefComponent;
