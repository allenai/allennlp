import React from 'react';


/*******************************************************************************
  <ModelIntro /> Component
*******************************************************************************/

class ModelIntro extends React.Component {
    render() {

      const { title, description } = this.props;

      return (
        <div>
          <h2>
            <span>{title}</span>
          </h2>
          <p>{description}</p>
        </div>
      );
  }
}

export default ModelIntro;
