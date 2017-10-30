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
      permalink: "waiting", // valid values: "waiting", null, {...data...}
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
    const slugRegex = /\/permalink\/([^/]+)\/?$/;
    const url = window.location.href;
    const match = slugRegex.exec(url);

    if (!match) {
      // This is not a permalink, so just render normally.
      this.setState({permalink: null});
    } else {
      const slug = match[1];
    }
  }

  render() {

    if (this.state.permalink == "waiting") {
      return (<div class="waiting-for-permalink">waiting for permalink</div>)
    } else if (this.state.permalink !== null) {
      return (<div class="rendered-permalink">rendered permalink</div>)
    }

    // Otherwise render the demo.
    const { selectedModel } = this.state;

    const ModelComponent = () => {
      if (selectedModel === "srl") {
        return (<SrlComponent />)
      }
      else if (selectedModel === "te") {
        return (<TeComponent />)
      }
      else if (selectedModel === "mc") {
        return (<McComponent />)
      }
    }

    return (
      <div className="pane-container">
        <Header selectedModel={selectedModel} changeModel={this.changeModel}/>
        <ModelComponent />
      </div>
    );
  }
}

export default App;
