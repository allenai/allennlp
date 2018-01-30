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

const parserExamples = [
    {
      table: "Season\tLevel\tDivision\tSection\tPosition\tMovements\n" +
             "1993\tTier 3\tDivision 2\tÖstra Svealand\t1st\tPromoted\n" +
             "1994\tTier 2\tDivision 1\tNorra\t11th\tRelegation Playoffs\n" +
             "1995\tTier 2\tDivision 1\tNorra\t4th\t\n" +
             "1996\tTier 2\tDivision 1\tNorra\t11th\tRelegation Playoffs - Relegated\n" +
             "1997\tTier 3\tDivision 2\tÖstra Svealand\t3rd\t\n" +
             "1998\tTier 3\tDivision 2\tÖstra Svealand\t7th\t\n" +
             "1999\tTier 3\tDivision 2\tÖstra Svealand\t3rd\t\n" +
             "2000\tTier 3\tDivision 2\tÖstra Svealand\t9th\t\n" +
             "2001\tTier 3\tDivision 2\tÖstra Svealand\t7th\t\n" +
             "2002\tTier 3\tDivision 2\tÖstra Svealand\t2nd\t\n" +
             "2003\tTier 3\tDivision 2\tÖstra Svealand\t3rd\t\n" +
             "2004\tTier 3\tDivision 2\tÖstra Svealand\t6th\t\n" +
             "2005\tTier 3\tDivision 2\tÖstra Svealand\t4th\tPromoted\n" +
             "2006*\tTier 3\tDivision 1\tNorra\t5th\t\n" +
             "2007\tTier 3\tDivision 1\tSödra\t14th\tRelegated",
      question: "What is the only year with the 1st position?",
    },
];

const title = "WikiTables Semantic Parsing";
const description = (
  <span>
    <span>
      Semantic parsing maps natural language to machine language.  This page demonstrates a semantic
      parsing model on the
      <a href="https://nlp.stanford.edu/software/sempre/wikitable/">WikiTableQuestions</a> dataset.
      The model is a re-implementation of the parser in the
      <a href="https://www.semanticscholar.org/paper/Neural-Semantic-Parsing-with-Type-Constraints-for-Krishnamurthy-Dasigi/8c6f58ed0ebf379858c0bbe02c53ee51b3eb398a">
      EMNLP 2017 paper by Krishnamurthy, Dasigi and Gardner</a>, which achieved state-of-the-art results
      on this dataset at the time.
    </span>
  </span>
);


class WikiTablesInput extends React.Component {
constructor(props) {
    super(props);

    // If we're showing a permalinked result,
    // we'll get passed in a table and question.
    const { table, question } = props;

    this.state = {
      tableValue: table || "",
      questionValue: question || ""
    };
    this.handleListChange = this.handleListChange.bind(this);
    this.handleQuestionChange = this.handleQuestionChange.bind(this);
    this.handleTableChange = this.handleTableChange.bind(this);
}

handleListChange(e) {
    if (e.target.value !== "") {
      this.setState({
          tableValue: parserExamples[e.target.value].table,
          questionValue: parserExamples[e.target.value].question,
      });
    }
}

handleTableChange(e) {
    this.setState({
      tableValue: e.target.value,
    });
}

handleQuestionChange(e) {
    this.setState({
    questionValue: e.target.value,
    });
}

render() {

    const { tableValue, questionValue } = this.state;
    const { outputState, runParser } = this.props;

    const parserInputs = {
    "tableValue": tableValue,
    "questionValue": questionValue
    };

    return (
        <div className="model__content">
        <ModelIntro title={title} description={description} />
            <div className="form__instructions"><span>Enter text or</span>
            <select disabled={outputState === "working"} onChange={this.handleListChange}>
                <option value="">Choose an example...</option>
                {parserExamples.map((example, index) => {
                  return (
                      <option value={index} key={index}>{example.table.substring(0,60) + "..."}</option>
                  );
                })}
            </select>
            </div>
            <div className="form__field">
            <label htmlFor="#input--mc-passage">Table</label>
            <textarea onChange={this.handleTableChange} id="input--mc-passage" type="text" required="true" autoFocus="true" placeholder="E.g. &quot;Season\tLevel\tDivision\tSection\tPosition\tMovements\n1993\tTier 3\tDivision 2\tÖstra Svealand\t1st\tPromoted\n1994\tTier 2\tDivision 1\tNorra\t11th\tRelegation Playoffs\n&quot;" value={tableValue} disabled={outputState === "working"}></textarea>
            </div>
            <div className="form__field">
            <label htmlFor="#input--mc-question">Question</label>
            <input onChange={this.handleQuestionChange} id="input--mc-question" type="text" required="true" value={questionValue} placeholder="E.g. &quot;What is the only year with the 1st position?&quot;" disabled={outputState === "working"} />
            </div>
            <div className="form__field form__field--btn">
            <Button enabled={outputState !== "working"} runModel={runParser} inputs={parserInputs} />
            </div>
        </div>
        );
    }
}


/*******************************************************************************
  <WikiTablesOutput /> Component
*******************************************************************************/

class WikiTablesOutput extends React.Component {
    render() {
      const { answer, logicalForm, actions, linking_scores, entities, question_tokens } = this.props;

      return (
        <div className="model__content">
          <div className="form__field">
            <label>Answer</label>
            <div className="model__content__summary">{ answer } Logical form execution (to actually get an answer) not yet supported</div>
          </div>

          <div className="form__field">
            <label>Logical Form</label>
            <div className="model__content__summary">{ logicalForm } Logical form recovery coming soon</div>
          </div>

          <div className="form__field">
            <Collapsible trigger="Model internals (beta)">
              <Collapsible trigger="Predicted actions">
                {actions.map((action, action_index) => (
                  <Collapsible key={"action_" + action_index} trigger={action['predicted_action']}>
                    <ActionInfo action={action} />
                  </Collapsible>
                ))}
              </Collapsible>
              <Collapsible trigger="Entity linking scores">
                  <HeatMap xLabels={question_tokens} yLabels={entities} data={linking_scores} xLabelWidth="250px" />
              </Collapsible>
            </Collapsible>
          </div>
        </div>
      );
    }
  }


class ActionInfo extends React.Component {
  render() {
    const { action } = this.props;
    const action_string = action['predicted_action'];
    const considered_actions = action['considered_actions'];
    const action_probs = action['action_probabilities'].map(x => [x]);

    return (
      <div className="heatmap">
        <HeatMap xLabels={['-']} yLabels={considered_actions} data={action_probs} xLabelWidth="250px" />
      </div>
    )
  }
}


/*******************************************************************************
  <McComponent /> Component
*******************************************************************************/

class _WikiTablesComponent extends React.Component {
    constructor(props) {
      super(props);

      const { requestData, responseData } = props;

      this.state = {
        outputState: responseData ? "received" : "empty", // valid values: "working", "empty", "received", "error"
        requestData: requestData,
        responseData: responseData
      };

      this.runParser = this.runParser.bind(this);
    }

    runParser(event, inputs) {
      this.setState({outputState: "working"});

      var payload = {
        table: inputs.tableValue,
        question: inputs.questionValue,
      };
      fetch(`${API_ROOT}/predict/wikitables-parser`, {
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
        const newPath = slug ? '/wikitables-parser/' + slug : '/wikitables-parser';

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

      const table = requestData && requestData.table;
      const question = requestData && requestData.question;
      const answer = responseData && responseData.answer;
      const logicalForm = responseData && responseData.logical_form;
      const actions = responseData && responseData.predicted_actions;
      const linking_scores = responseData && responseData.linking_scores;
      const entities = responseData && responseData.entities;
      const question_tokens = responseData && responseData.question_tokens;

      return (
        <div className="pane model">
          <PaneLeft>
            <WikiTablesInput runParser={this.runParser}
                             outputState={this.state.outputState}
                             table={table}
                             question={question}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <WikiTablesOutput answer={answer}
                              logicalForm={logicalForm}
                              actions={actions}
                              linking_scores={linking_scores}
                              entities={entities}
                              question_tokens={question_tokens}
            />
          </PaneRight>
        </div>
      );

    }
}

const WikiTablesComponent = withRouter(_WikiTablesComponent);

export default WikiTablesComponent;
