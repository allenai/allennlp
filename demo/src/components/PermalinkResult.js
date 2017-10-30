import React from 'react';


/*******************************************************************************
  <PermalinkResult /> Component
*******************************************************************************/

class PermalinkResult extends React.Component {
    render() {

      const { permadata } = this.props;
      console.log(permadata);
      const { modelName, requestData, responseData } = permadata;

      return (
        <div>
          <h2>
            <span>{modelName}</span>
          </h2>
          <p>{JSON.stringify(requestData)}</p>
          <p>{JSON.stringify(responseData)}</p>
        </div>
      );
  }
}

export default PermalinkResult;
