Plot2d = function(container) {
  this.container = container;
  this.svg = d3.select(this.container).html('').append('svg');
  this._marginTop = 20;
  this._marginRight = 20;
  this._marginBottom = 50;
  this._marginLeft = 70;
  var containerPaddingLeft = window.getComputedStyle(this.container, null).getPropertyValue('padding-left');
  var containerPaddingRight = window.getComputedStyle(this.container, null).getPropertyValue('padding-right');
  containerPaddingLeft = +(containerPaddingLeft.substr(0, containerPaddingLeft.length - 2));
  containerPaddingRight = +(containerPaddingRight.substr(0, containerPaddingRight.length - 2));
  this._width = +this.container.clientWidth - (containerPaddingLeft + containerPaddingRight);
  this._height = (+this._width)*3/4;
  this._type = 'scatter';
  this._xAxisScale = 'Linear';
  this._yAxisScale = 'Linear';
  this._xAxisTitle = 'X';
  this._yAxisTitle = 'Y';
};
Plot2d.prototype.data = function(data) {
  this._data = data;
  return this;
};
Plot2d.prototype.type = function(type) {
  this._type = type;
  return this;
};
Plot2d.prototype.xAxisScale = function(scale) {
  this._xAxisScale = scale;
  return this;
};
Plot2d.prototype.xAxisTitle = function(title) {
  this._xAxisTitle = title;
  return this;
};
Plot2d.prototype.yAxisScale = function(scale) {
  this._yAxisScale = scale;
  return this;
};
Plot2d.prototype.yAxisTitle = function(title) {
  this._yAxisTitle = title;
  return this;
};
Plot2d.prototype.draw = function(xName, yName) {
  var self = this;
  var svg = this.svg;
  svg.attr('width', this._width)
     .attr('height', this._height);
  var svg = svg.append('g').attr('transform', 'translate(' + this._marginLeft + ',' + this._marginTop + ')');
  var tooltip = d3.select(this.container).append('div')
                                         .attr('class', 'tooltip bs-tooltip-top')
                                         .style('opacity', 0);
  tooltip.append('div').attr('class', 'arrow');
  var tooltipInner = tooltip.append('div').attr('class', 'tooltip-inner');
  var width = this._width - (this._marginLeft + this._marginRight);
  var height = this._height - (this._marginTop + this._marginBottom);
  // Set X-axis
  var x;
  switch (this._type) {
  case 'bar':
    x = d3.scaleBand().range([0, width]).padding(0.1);
    x.domain(this._data.map(function(d) { return d[xName]; }));
    if (this._xAxisScale != 'Linear') {
      throw new Error('unsupported x-axis type in bar plot: '+this._xAxisScale);
    }
    break;
  case 'scatter':
    x = d3['scale'+this._xAxisScale]().range([0, width]);
    x.domain(d3.extent(this._data, function(d) { return d[xName]; })).nice();
    break;
  default:
    throw new Error('unknown type: '+this._type);
  }
  svg.append('g').attr('class', 'axis')
                 .attr('transform', 'translate(0,' + height + ')')
                 .call(d3.axisBottom(x));
  svg.append('text').attr('class', 'axis-title')
                    .attr('transform', 'translate(' + (width/2) + ',' + (height + this._marginTop + 20) + ')')
                    .style('text-anchor', 'middle')
                    .text(this._xAxisTitle);
  // Set Y-axis
  var y = d3['scale'+this._yAxisScale]().range([height, 0]);
  y.domain(d3.extent(this._data, function(d) { return d[yName]; })).nice();
  svg.append('g').attr('class', 'axis')
                 .call(d3.axisLeft(y));
  svg.append('text').attr('class', 'axis-title')
                    .attr('transform', 'rotate(-90)')
                    .attr('x', -height/2)
                    .attr('y', -this._marginLeft)
                    .attr('dy', '1em')
                    .style('text-anchor', 'middle')
                    .text(this._yAxisTitle);
  // Set points
  switch (this._type) {
  case 'bar':
    svg.selectAll('.bar').data(this._data)
       .enter().append('rect').attr('class', 'bar')
                              .attr('x', function(d) { return x(d[xName]); })
                              .attr('width', x.bandwidth())
                              .attr('y', function(d) { return y(d[yName]); })
                              .attr('height', function(d) { return height - y(d[yName]); })
                              .on('mouseover', function(d) {
                                tooltip.transition().duration(250).style("opacity", 1.0);
                                tooltipInner.text('('+d[xName]+', '+d[yName]+')');
                                tooltip.style('left', (x(d[xName]) + self._marginLeft + x.bandwidth()/2 - tooltipInner.node().getBoundingClientRect().width/2) + "px")
                                       .style('top', y(d[yName]) + "px");
                                d3.select(this).classed('bar-mouseover', true);
                              })
                              .on('mouseout', function(d) {
                                tooltip.transition().duration(1000).style("opacity", 0);
                                d3.select(this).classed('bar-mouseover', false);
                              });
    break;
  case 'scatter':
    svg.selectAll('.point').data(this._data)
       .enter().append('circle').attr('class', 'point')
                                .attr('r', 3)
                                .attr('cx', function(d) { return x(d[xName]); })
                                .attr('cy', function(d) { return y(d[yName]); });
    break;
  }
};
function plot2d(container) {
  return new Plot2d(container);
}
