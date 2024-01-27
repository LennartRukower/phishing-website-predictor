import { useEffect, useRef } from "react";
import * as d3 from "d3";

const ParallelCoordinatesChart = ({ listOfTrainingData, listOfCurrData }) => {
    const d3Container = useRef(null);

    useEffect(() => {
        if (listOfTrainingData && listOfCurrData && d3Container.current) {
            const margin = { top: 30, right: 10, bottom: 10, left: 0 },
                width = 2000 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom;

            // Calculate the viewbox by subtracting and adding some padding from the width and height
            const viewBoxX = -margin.left - 15; // 15 pixels padding on the left
            const viewBoxY = -margin.top;
            const viewBoxWidth = width + margin.left + margin.right + 30; // Add 30 pixels padding total (15 left + 15 right)
            const viewBoxHeight = height + margin.top + margin.bottom;

            d3.select(d3Container.current).selectAll("*").remove();
            const svg = d3
                .select(d3Container.current)
                .append("svg")
                .attr("width", width + margin.left + margin.right)
                .attr("height", height + margin.top + margin.bottom)
                .attr("viewBox", `${viewBoxX} ${viewBoxY} ${viewBoxWidth} ${viewBoxHeight}`)
                .append("g")
                .attr("transform", `translate(${margin.left},${margin.top})`);

            // Extract the list of dimensions and create a scale for each
            const dimensions = Object.keys(listOfTrainingData[0]).filter((d) => d !== "label");
            const x = d3.scalePoint().range([0, width]).domain(dimensions);

            // For each dimension, create a linear scale
            const y = {};
            for (const dim of dimensions) {
                y[dim] = d3
                    .scaleLinear()
                    .domain(d3.extent(listOfTrainingData, (d) => +d[dim]))
                    .range([height, 0]);
            }

            // Draw the lines
            const color = d3.scaleOrdinal(d3.schemeCategory10);

            svg.selectAll("myPath")
                .data(listOfTrainingData)
                .enter()
                .append("path")
                .attr("d", function (d) {
                    return d3.line()(dimensions.map((p) => [x(p), y[p](d[p])]));
                })
                .style("fill", "none")
                .style("stroke", (d) => color(d.label))
                .style("opacity", 0.5);

            // Draw the line for current data
            svg.append("path")
                .datum(listOfCurrData)
                .attr("d", (d) => d3.line()(dimensions.map((p) => [x(p), y[p](d[p])])))
                .style("fill", "none")
                .style("stroke", "black") // Change to desired color for current data
                .style("stroke-width", "2px");

            // Add the axes
            for (const dim of dimensions) {
                svg.append("g")
                    .attr("transform", `translate(${x(dim)}, 0)`)
                    .call(d3.axisLeft(y[dim]))
                    .append("text")
                    .attr("y", margin.left * -1) // Adjust this value to position your label appropriately
                    .style("text-anchor", "middle")
                    .attr("y", -9)
                    .text(dim)
                    .style("fill", "black");
            }
        }
    }, [listOfTrainingData, listOfCurrData]); // Redraw chart if data changes

    return (
        <div className="w-full overflow-x-scroll">
            <div ref={d3Container}></div>;
        </div>
    );
};

export default ParallelCoordinatesChart;
