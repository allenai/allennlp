import React from 'react';
import PropTypes from 'prop-types';

const FixedBox = ({children, width}) => {
  return <div style={{flex: `0 0 ${width}px`}}> {children} </div>;
};

FixedBox.defaultProps = {
  children: ' ',
};

FixedBox.propTypes = {
  children: PropTypes.oneOfType([PropTypes.string, PropTypes.element]),
  width: PropTypes.number.isRequired,
};

export default FixedBox;
