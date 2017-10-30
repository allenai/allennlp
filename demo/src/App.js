import React from 'react';
import SrlComponent from './components/SrlComponent';
import TeComponent from './components/TeComponent';
import McComponent from './components/McComponent';
import Header from './components/Header';

/*******************************************************************************
  <App /> Container
*******************************************************************************/

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      permalink: "waiting", // valid values: "url", null, {...data...}
      selectedModel: "srl", // valid values: "srl", "mc", "te"
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
    // Check if this is a /permalink/xyz URL. If so, grab the slug
    const slugRegex = /\/permalink\/([^/]+)\/?$/;
    const url = window.location.href;
    const match = slugRegex.exec(url);

    if (!match) {
      // This is not a permalink, so just render normally.
      this.setState({permalink: null});
      console.log(this.state);
    } else {
      const slug = match[1];
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
        this.setState({permalink: json, selectedModel: json.modelName});
      }).catch((error) => {
        this.setState({outputState: "error"});
        throw error; // todo(michaels): is this right?
      });
    }
  }

  render() {
    const permadata = this.state.permalink;
    const { selectedModel } = this.state;

    const ModelComponent = () => {
      if (selectedModel === "srl") {
        return (<SrlComponent permadata={permadata}/>)
      }
      else if (selectedModel === "te") {
        return (<TeComponent permadata={permadata}/>)
      }
      else if (selectedModel === "mc") {
        return (<McComponent permadata={permadata}/>)
      }
    }

    if (this.state.permalink === null) {
      // Not a permalink, so render the demo

      return (
        <div className="pane-container">
          <Header selectedModel={selectedModel} changeModel={this.changeModel}/>
          <ModelComponent />
        </div>
      );
    } else {
      // It is a permalink, so render using that path:
      return (
        <div className="pane-container">
          <Header permalink={true}/>
          <ModelComponent />
        </div>
      )
    }
  }
}

export default App;
