import React from 'react';
import PropTypes from 'prop-types';
import FixedBox from './FixedBox';

const DataGrid = ({
  xLabels,
  yLabels,
  data,
  xLabelWidth,
  background,
  height,
}) => {
  const flatArray = data.reduce((i, o) => [...o, ...i], []);
  const max = Math.max(...flatArray);
  const min = Math.min(...flatArray);
  return (
    <div>
      {yLabels.map((y, yi) => (
        <div key={`${y}_${yi}`} style={{display: 'flex'}}>
          <FixedBox width={xLabelWidth}>
            <div style={{textAlign: 'right', paddingRight: '5px', paddingTop:`${height/3.7}px`}}>{y}</div>
          </FixedBox>
          {xLabels.map((x, xi) => (
            <div
              title={`${data[yi][xi]}`}
              key={`${x}_${xi}_${y}_${yi}`}
              style={{
                background,
                margin: '1px 1px 0 0',
                height,
                flex: 1,
                opacity: (data[yi][xi] - min) / (max - min),
              }}
            >
              &nbsp;
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};

DataGrid.propTypes = {
  xLabels: PropTypes.arrayOf(
    PropTypes.oneOfType([PropTypes.string, PropTypes.number])
  ).isRequired,
  yLabels: PropTypes.arrayOf(
    PropTypes.oneOfType([PropTypes.string, PropTypes.number])
  ).isRequired,
  data: PropTypes.arrayOf(PropTypes.array).isRequired,
  background: PropTypes.string.isRequired,
  height: PropTypes.number.isRequired,
  xLabelWidth: PropTypes.number.isRequired,
};

export default DataGrid;
