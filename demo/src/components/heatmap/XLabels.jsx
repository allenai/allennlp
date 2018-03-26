/*
 * The column labels for the `HeatMap` component.  We rotate the labels 90 degrees, so we maintain
 * an even-sided grid.
 */
import React from 'react';

function XLabels({labels, width, boxSize}) {
  return (
    <div>
      <div style={{display: "inline-block", width: width, height: "4em"}} />
      {labels.map((x, xi) => (
        <div
          key={`${x}_${xi}`}
          style={{display: "inline-block",
                  transform: "rotate(-90deg)",
                  margin: "1px 1px 0 0",
                  width: boxSize,
                  textAlign: 'center'}}
        >
          {x}
        </div>
      ))}
    </div>
  );
}

export default XLabels;
