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

  render() {

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
