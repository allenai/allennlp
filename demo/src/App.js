import React from 'react';
import { API_ROOT } from './api-config';
import { BrowserRouter as Router, Route, Redirect } from 'react-router-dom';
import SrlComponent from './components/SrlComponent';
import TeComponent from './components/TeComponent';
import McComponent from './components/McComponent';
import CorefComponent from './components/CorefComponent'
import NamedEntityComponent from './components/NamedEntityComponent'
import ConstituencyParserComponent from './components/ConstituencyParserComponent'
import Menu from './components/Menu';
import ModelIntro from './components/ModelIntro'
import { PaneTop } from './components/Pane'
import WaitingForPermalink from './components/WaitingForPermalink'

/*******************************************************************************
  <App /> Container
*******************************************************************************/

const DEFAULT_PATH = "/machine-comprehension"

// The App is just a react-router wrapped around the Demo component.
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

    // React router supplies us with a model name and (possibly) a slug.
    const { model, slug } = props.match.params;

    this.state = {
      slug: slug,
      selectedModel: model,
      requestData: null,
      responseData: null
    };

    // We'll need to pass this to the Header component so that it can clear
    // out the data when you switch from one model to another.
    this.clearData = () => {
      this.setState({requestData: null, responseData: null})
    }

    // Our components will be using history.push to change the location,
    // and they will be attaching any `requestData` and `responseData` updates
    // to the location object. That means we need to listen for location changes
    // and update our state accordingly.
    props.history.listen((location, action) => {
      const { state } = location;
      if (state) {
        const { requestData, responseData } = state;
        this.setState({requestData, responseData})
      }
    });
  }

  // We also need to update the state whenever we receive new props from React router.
  componentWillReceiveProps({ match }) {
    const { model, slug } = match.params;
    this.setState({selectedModel: model, slug: slug});
  }

  // After the component mounts, we check if we need to fetch the data
  // for a permalink.
  componentDidMount() {
    const { slug, responseData } = this.state;

    // If this is a permalink and we don't yet have the data for it...
    if (slug && !responseData) {
      // Make an ajax call to get the permadata,
      // and then use it to update the state.
      fetch(`${API_ROOT}/permadata`, {
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
        this.setState({requestData, responseData});
      }).catch((error) => {
        this.setState({outputState: "error"});
        console.error(error);
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
      else if (selectedModel === "coreference-resolution") {
        return (<CorefComponent requestData={requestData} responseData={responseData}/>)
      }
      else if (selectedModel === "named-entity-recognition") {
        return (<NamedEntityComponent requestData={requestData} responseData={responseData}/>)
      }
      else if (selectedModel === "constituency-parsing") {
        return (<ConstituencyParserComponent requestData={requestData} responseData={responseData}/>)
      }
      else if (selectedModel === "user-models") {
        const modelRequest = "User Contributed Models"
        const modelDescription = (
          <span>
            <span>
              AllenNLP is looking to add contributed models implemented using AllenNLP as either library components or demos (with free hosting!).
              If you have a published result or novel model demonstrating strong performance on a dataset and you are interested
              in adding your model to a list of publically available implementations, as a service to this demo, or as a component in the library itself,
              please consider opening an issue on our
            </span>
            <a href="https://github.com/allenai/allennlp/issues" target="_blank" rel="noopener noreferrer">{' '} public Github repository </a>
            <span>
              or sending us an email at allennlp-contact@allenai.org to discuss what you have in mind.
            </span>
          </span>
      );

        return (
          <div className="pane__horizontal model">
            <div className='model__content'>
              <PaneTop>
                <ModelIntro title={modelRequest} description={modelDescription}/>
              </PaneTop>
              </div>
          </div>

        )
      }
    }

    return (
      <div className="pane-container">
        <Menu selectedModel={selectedModel} clearData={this.clearData}/>
        <ModelComponent />
      </div>
    );
  }
}

export default App;
