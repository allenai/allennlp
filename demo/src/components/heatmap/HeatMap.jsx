/*
 * A scrollable component that displays a two-dimensional heat map, with labels on the rows and the
 * columns.  This is done with a series of divs, where the first div has the column labels (the
 * `XLabels` component) and the rest of the divs each have a row label and then colored boxes for
 * each cell, where the colors are determined by the data values (the `DataGrid` component).
 */
import React from 'react';
import PropTypes from 'prop-types';
import XLabels from './XLabels';
import DataGrid from './DataGrid';

function HeatMap({xLabels, yLabels, data, background, height, xLabelWidth, boxSize}) {
  return (
    <div style={{height: height, overflow: "scroll", "overflow-x": "scroll", "white-space": "nowrap"}}>
      <XLabels labels={xLabels} width={xLabelWidth} boxSize={boxSize} />
      <DataGrid
        {...{xLabels, yLabels, data, background, height, xLabelWidth, boxSize}}
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
  height: 600,
  xLabelWidth: 60,
  boxSize: 28,
  overflow: "scroll",
};

export default HeatMap;
