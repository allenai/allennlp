import React from 'react';
import { BrowserRouter as Router, Route, Redirect } from 'react-router-dom';
import SrlComponent from './components/SrlComponent';
import TeComponent from './components/TeComponent';
import McComponent from './components/McComponent';
import Header from './components/Header';
import WaitingForPermalink from './components/WaitingForPermalink'

/*******************************************************************************
  <App /> Container
*******************************************************************************/

const DEFAULT_PATH = "/semantic-role-labeling"

const App = () => (
  <Router>
    <div>

      <Route exact path="/" render={() => (
        <Redirect to={DEFAULT_PATH}/>
      )}/>
      <Route path="/:model/:slug?" component={Demo}/>
    </div>
  </Router>
)

class Demo extends React.Component {
  constructor(props) {
    super(props);

    const { model, slug } = props.match.params;

    this.state = {
      slug: slug,
      selectedModel: model,
      requestData: null,
      responseData: null
    };

    this.clearData = () => {
      this.setState({requestData: null, responseData: null})
    }

    props.history.listen((location, action) => {
      console.log(location);
      console.log(action);
      const { state } = location;
      if (state) {
        const { requestData, responseData } = state;
        this.setState({requestData: requestData, responseData: responseData})
      }
    });
  }

  componentWillReceiveProps({ match }) {
    const { model, slug } = match.params;

    // Only trigger setState if this is an actual change.
    if (model !== this.state.model || slug !== this.state.slug) {
      this.setState({selectedModel: model, slug: slug});
    }
  }

  componentDidMount() {
    const { slug } = this.state;
    const { location } = this.props;

    // If this is a permalink and we don't yet have the data for it...
    if (slug && !location.responseData) {
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
        const { requestData, responseData } = json;
        this.setState({requestData: requestData, responseData: responseData});
      }).catch((error) => {
        this.setState({outputState: "error"});
        throw error; // todo(michaels): is this right?
      });
    }
  }

  render() {
    const { slug, selectedModel, requestData, responseData } = this.state;

    const ModelComponent = () => {
      if (slug && !responseData) {
        // We're still waiting for permalink data, so just return the placeholder component.
        return (<WaitingForPermalink/>)
      } else if (selectedModel === "semantic-role-labeling") {
        return (<SrlComponent requestData={requestData} responseData={responseData}/>)
      }
      else if (selectedModel === "textual-entailment") {
        return (<TeComponent requestData={requestData} responseData={responseData}/>)
      }
      else if (selectedModel === "machine-comprehension") {
        return (<McComponent requestData={requestData} responseData={responseData}/>)
      }
    }

    return (
      <div className="pane-container">
        <Header selectedModel={selectedModel} clearData={this.clearData}/>
        <ModelComponent />
      </div>
    );
  }
}

export default App;
