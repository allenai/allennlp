import React from 'react';
import PropTypes from 'prop-types';
import XLabels from './XLabels';
import DataGrid from './DataGrid';

function HeatMap({xLabels, yLabels, data, background, height, xLabelWidth}) {
  return (
    <div>
      <XLabels labels={xLabels} width={xLabelWidth} />
      <DataGrid
        {...{xLabels, yLabels, data, background, height, xLabelWidth}}
      />
    </div>
  );
}

HeatMap.propTypes = {
  xLabels: PropTypes.arrayOf(
    PropTypes.oneOfType([PropTypes.string, PropTypes.number])
  ).isRequired,
  yLabels: PropTypes.arrayOf(
    PropTypes.oneOfType([PropTypes.string, PropTypes.number])
  ).isRequired,
  data: PropTypes.arrayOf(PropTypes.array).isRequired,
  background: PropTypes.string,
  height: PropTypes.number,
  xLabelWidth: PropTypes.number,
};

HeatMap.defaultProps = {
  background: '#329fff',
  height: 30,
  xLabelWidth: 60,
};

export default HeatMap;
