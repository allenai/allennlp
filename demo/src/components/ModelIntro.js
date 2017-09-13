import React from 'react';


/*******************************************************************************
  <ModelIntro /> Component
*******************************************************************************/

class ModelIntro extends React.Component {
    render() {

      const { title, tooltip } = this.props;

      return (
        <h2>
          <span>{title}</span>
          <div className="tooltip">
            <svg className="tooltip__trigger">
              <use xlinkHref="#icon__help"></use>
            </svg>
            <div className="tooltip__cursor-container">
              <div className="tooltip__box">{tooltip}</div>
            </div>
          </div>
        </h2>
      );
    }
  }

  export default ModelIntro;
