import React from 'react';


/*******************************************************************************
  <ResultDisplay /> Component
*******************************************************************************/

class ResultDisplay extends React.Component {

    render() {
      const { outputState } = this.props;

      const placeholderTemplate = (message) => {
        return (
          <div className="placeholder">
            <div className="placeholder__content">
              <svg className={`placeholder__${outputState}`}>
                <use xlinkHref={`#icon__${outputState}`}></use>
              </svg>
              {message !== "" ? (
                <p>{message}</p>
              ) : null}
            </div>
          </div>
        );
      }

      let outputContent;
      switch (outputState) {
        case "working":
          outputContent = placeholderTemplate("");
          break;
        case "received":
          outputContent = this.props.children;
          break;
        case "error":
          outputContent = placeholderTemplate("Something went wrong. Please try again.");
          break;
        default:
          // outputState = "empty"
          outputContent = placeholderTemplate("Run model to view results");
      }

      return (
        <div className={`pane__right model__output ${outputState !== "received" ? "model__output--empty" : ""}`}>
          <div className="pane__thumb"></div>
          {outputContent}
        </div>
      );
    }
}


/*******************************************************************************
  <PaneRight /> Component
*******************************************************************************/

export class PaneRight extends React.Component {
    render() {
      const { outputState } = this.props;

      return (
        <ResultDisplay outputState={outputState}>
          {this.props.children}
        </ResultDisplay>
      )
    }
}

/*******************************************************************************
<PaneLeft /> Component
*******************************************************************************/

export class PaneLeft extends React.Component {

    render () {
      return (
        <div className="pane__left model__input">
          {this.props.children}
        </div>
      );
    }
}
