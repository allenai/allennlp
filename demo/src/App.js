import React from 'react';
import SrlComponent from './components/SrlComponent';
import TeComponent from './components/TeComponent';
import McComponent from './components/McComponent';
import Header from './components/Header';
import WaitingForPermalink from './components/WaitingForPermalink'

/*******************************************************************************
  <App /> Container
*******************************************************************************/

class App extends React.Component {
  constructor() {
    super();

    // Check if this is a /permalink/xyz URL.
    // This regex will capture the slug if it matches.
    const slugRegex = /\/permalink\/([^/]+)\/?$/;
    const url = window.location.href;
    const match = slugRegex.exec(url);

    this.state = {
      match: match,
      permadata: null,
      selectedModel: "semantic-role-labeling",
      rawOutput: "",
    };
    this.changeModel = this.changeModel.bind(this);
  }

  changeModel(thisModel) {
    this.setState({
      selectedModel: thisModel,
    });
  }

  componentDidMount() {
    const { match } = this.state;

    if (match) {
      // match[0] is "/permalink/xyz"
      // match[1] is "xyz"
      const slug = match[1];

      // Make an ajax call to get the permadata,
      // and then use it to update the state.
      fetch('http://localhost:8000/permadata', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({"slug": slug})
      }).then(function(response) {
        return response.json();
      }).then((json) => {
        this.setState({permadata: json, selectedModel: json.modelName});
      }).catch((error) => {
        this.setState({outputState: "error"});
        throw error; // todo(michaels): is this right?
      });
    }
  }

  render() {
    const { match, permadata, selectedModel } = this.state;

    const HeaderComponent = () => {
      if (match) {
        // This is a permalink request, so return that header.
        return (<Header permalink={true}/>)
      } else {
        // Otherwise return the usual header.
        return (<Header selectedModel={selectedModel} changeModel={this.changeModel}/>)
      }
    }

    const ModelComponent = () => {
      if (match && !permadata) {
        // We're still waiting for permalink data, so just return the placeholder component.
        return (<WaitingForPermalink/>)
      } else if (selectedModel === "semantic-role-labeling") {
        return (<SrlComponent permadata={permadata}/>)
      }
      else if (selectedModel === "textual-entailment") {
        return (<TeComponent permadata={permadata}/>)
      }
      else if (selectedModel === "machine-comprehension") {
        return (<McComponent permadata={permadata}/>)
      }
    }

    return (
      <div className="pane-container">
        <HeaderComponent/>
        <ModelComponent />
      </div>
    );
  }
}

export default App;
