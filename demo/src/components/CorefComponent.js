import React from 'react';
import { API_ROOT } from '../api-config';
import { withRouter } from 'react-router-dom'
import {PaneLeft, PaneRight} from './Pane'
import Button from './Button'
import ModelIntro from './ModelIntro'


/*******************************************************************************
  <CorefInput /> Component
*******************************************************************************/

const corefExamples = [
    {
      document: "We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes ."
    },
    {
      document: "Xuming Zhang , Chairman of the Chinese Enterprise Association in Macau said that , at present there were more than 200 enterprises operating with Chinese capital in Macau and that the total value of assets is more than 90 billion patacas . Chinese capital enterprises have become the biggest foreign investors in Macau . Xuming Zhang recently said at the joint meeting for the fifth anniversary of the establishment of the Chinese Enterprise Association in Macau , that Macau 's inland investment enterprises , from small to large and from weak to strong , have developed into an important force in Macau 's economic domain .  They have made important contributions to the prosperity and stability of Macau . According to presentations , these enterprises have extensively taken part in many areas of operating activities such as trade , industry , finance , insurance , tourism , catering , traffic and transportation , construction , real estate , etc. in Macau . Among these , the proportion that Chinese capital accounts for in financial insurance has reached 50 % . It accounts for from 50 % to 70 % of the tourism industry , accounts for 30 % of imports and exports , and accounts for 70 % of real estate . Xuming Zhang expressed that enterprises operating with Chinese capital in Macau will continue to take the direction of Xiaoping Deng 's program of  one country , two systems  and of all national guidelines and policies for Hong Kong and Macau , to adhere to the principle of  Some to do and some not to do , and to strive together with local figures in industrial and commercial circles to make more contributions to promoting Macau 's economic stability and social development .",
    },
    {
      document: "Friday's 190 - point plunge in stocks does not come atop the climate of anxiety that dominated financial markets just prior to their 1987 October crash , and mechanisms have been put in place to keep markets more orderly . Still , the lesson is about the same : On Friday the 13th , the market was spooked by Washington. The consensus along the street seems to be that the plunge was triggered by the financing problems of the UAL takeover, and it's certainly true the rout began immediately after the UAL trading halt. Still, the consensus seems almost as wide that one faltering bid is no reason to write down the value of all U.S. business. This observation leads us to another piece of news moving on the Dow Jones ticker shortly before the downturn: the success of Senate Democrats in stalling the capital gains tax cut. The real value of all shares , after all , is directly impacted by the tax on any profits (all the more so given the limits on deductions for losses that show gains are not  'ordinary income'). And market expectations clearly have been raised by the capital gains victory in the House last month. An hour before Friday 's plunge , that provision was stripped from the tax bill, leaving it with $5.4 billion in tax increases without a capital gains cut . There is a great deal to be said , to be sure , for stripping the garbage out of the reconciliation bill . It would be a good thing if Congress started to decide issues one - by - one on their individual merits without trickery. For one thing, no one doubts that the capital gains cut would pass on an up-or-down vote . Since Senate leaders have so far fogged it up with procedural smokescreens , promises of a cleaner bill are suspect . Especially so since President Bush has been weakened by the Panama fiasco. To the extent that the UAL troubles contributed to the plunge , they are another instance of Washington 's sticky fingers. As the best opportunities for corporate restructurings are exhausted of course, at some point the market will start to reject them . But the airlines are scarcely a clear case , given anti-takeover mischief by Secretary of Transportation Skinner , who professes to believe safety will be compromised if KLM and British Airways own interests in companies that fly airplanes . Worse , Congress has started to jump on the Skinner bandwagon . James Oberstar , the Minnesota Democrat who chairs the Public Works and Transportation Committee 's aviation subcommittee , has put an anti-airline takeover bill on supersonic speed so that it would be passed in time to affect the American and United Air Lines bids . It would give Mr. Skinner up to 50 days to `` review '' any bid for 15% or more of the voting stock of any U.S. carrier with revenues of $1 billion or more . So the UAL deal has problems , and the market loses 190 points . Congratulations , Mr. Secretary and Mr. Congressman . In the 1987 crash , remember , the market was shaken by a Danny Rostenkowski proposal to tax takeovers out of existance . Even more important , in our view , was the Treasury 's threat to thrash the dollar . The Treasury is doing the same thing today ; thankfully , the dollar is not under 1987 - style pressure . Also , traders are in better shape today than in 1987 to survive selling binges . They are better capitalized . They are in less danger of losing liquidity simply because of tape lags and clearing and settlement delays . The Fed promises any needed liquidity . The Big Board 's liaison with the Chicago Board of Trade has improved; it will be interesting to learn if  'circuit breakers' prove to be a good idea . In any event , some traders see stocks as underpriced today , unlike 1987. There is nothing wrong with the market that can't be cured by a little coherence and common sense in Washington . But on the bearish side , that may be too much to expect. First Chicago Corp. posted a third - quarter loss of $23.3 million after joining other big banks in further adding to its reserves for losses on foreign loans . The parent company of First National Bank of Chicago, with $48 billion in assets , said it set aside $200 million to absorb losses on loans and investments in financially troubled countries. The addition, on top of two big 1987 additions to foreign-loan reserves , brings the reserve to a level equaling 79% of medium-term and long-term loans outstanding to troubled nations. First Chicago since 1987 has reduced its loans to such nations to $1.7 billion from $3 billion. Despite this loss, First Chicago said it doesn't need to sell stock to raise capital. ",
    }
];

const title = "Co-reference Resolution";
const description = (
  <span>
    <span>
    Coreference resolution is the task of finding all expressions that refer to the same entity
    in a text. It is an important step for a lot of higher level NLP tasks that involve natural
    language understanding such as document summarization, question answering, and information extraction.
    </span>
    <a href = "https://www.semanticscholar.org/paper/End-to-end-Neural-Coreference-Resolution-Lee-He/3f2114893dc44eacac951f148fbff142ca200e83" target="_blank" rel="noopener noreferrer">{' '} End-to-end Neural Coreference Resolution ( Lee et al, 2017) {' '}</a>
    <span>
    is a neural model which considers all possible spans in the document as potential mentions and
    learns distributions over possible anteceedents for each span, using aggressive, learnt
    pruning strategies to retain computational efficiency. It achieved state-of-the-art accuracies on
    </span>
    <a href = "http://cemantix.org/data/ontonotes.html" target="_blank" rel="noopener noreferrer">{' '} the Ontonotes 5.0 dataset {' '}</a>
    <span>
    in early 2017.
    </span>
  </span>
);


class CorefInput extends React.Component {
    constructor(props) {
        super(props);

        // If we're showing a permalinked result, we'll get passed in a document.
        const { doc } = props;

        this.state = {
        corefDocumentValue: doc || "",
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
                required="true" autoFocus="true" placeholder="We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes . For some , an awareness of this exit strategy permeates the enterprise , allowing them to skimp on the niceties they would more or less have to extend toward a person they were likely to meet again ." value={corefDocumentValue} disabled={outputState === "working"}></textarea>
                </div>
                <div className="form__field form__field--btn">
                <Button enabled={outputState !== "working"} outputState={outputState} runModel={runCorefModel} inputs={corefInputs} />
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
      const { doc, clusters } = this.props;
      var clusteredDocument = doc.map((word, wordIndex) => {
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
          return "coref__span";
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
               <li key={ index }>
                {cluster.map((span, wordIndex) =>
                  <a key={ wordIndex } onMouseEnter={ () => this.onClusterMouseover(index) }> {doc.slice(span[0], span[1] + 1).join(" ")},</a>
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
              <span key={ index } className={ wordStyle(clusteredWord) }> {clusteredWord['word']}</span>
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

class _CorefComponent extends React.Component {
    constructor(props) {
      super(props);

      const { requestData, responseData } = props;

      this.state = {
        requestData: requestData,
        responseData: responseData,
        outputState: responseData ? "received" : "empty" // valid values: "working", "empty", "received", "error"
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

      fetch(`${API_ROOT}/predict/coreference-resolution`, {
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
        const newPath = slug ? '/coreference-resolution/' + slug : '/coreference-resolution';

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

      const inputDoc = requestData && requestData.document;
      const outputDoc = responseData && responseData.document;
      const clusters = responseData && responseData.clusters;

      return (
        <div className="pane model">
          <PaneLeft>
            <CorefInput runCorefModel={this.runCorefModel} outputState={this.state.outputState} doc={inputDoc}/>
          </PaneLeft>
          <PaneRight outputState={this.state.outputState}>
            <CorefOutput doc={outputDoc} clusters={clusters}/>
          </PaneRight>
        </div>
      );
    }
}

const CorefComponent = withRouter(_CorefComponent)

export default CorefComponent;
