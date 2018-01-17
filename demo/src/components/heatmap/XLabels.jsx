import React from 'react';
import PropTypes from 'prop-types';
import FixedBox from './FixedBox'

function XLabels({labels, width}) {
  return (
    <div style={{display: 'flex'}}>
      <FixedBox width={width} />
      {labels.map(x => (
        <div key={x} style={{flex: 1, textAlign: 'center'}}>
          {x}
        </div>
      ))}
    </div>
  );
}

export default XLabels;
