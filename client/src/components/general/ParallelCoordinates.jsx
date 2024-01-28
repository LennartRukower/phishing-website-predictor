import { useEffect, useRef } from "react";
import * as d3 from "d3";

const ParallelCoordinatesChart = ({ trainingData, currentData }) => {
    const d3Container = useRef(null);

    useEffect(() => {
        if (trainingData && currentData && d3Container.current) {
            const margin = { top: 30, right: 10, bottom: 100, left: 0 },
                width = trainingData.length * 24 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom;
            const viewBoxX = -margin.left - 25;
            const viewBoxY = -margin.top;
            const viewBoxWidth = width + margin.left + margin.right + 40;
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
            const dimensions = Object.keys(trainingData[0]).filter((d) => d !== "label");
            const x = d3.scalePoint().range([0, width]).domain(dimensions);

            // For each dimension, create a linear scale
            const y = {};
            for (const dim of dimensions) {
                y[dim] = d3
                    .scaleLinear()
                    .domain(d3.extent([...trainingData, currentData], (d) => +d[dim]))
                    .range([height, 0]);
            }

            // Draw the lines
            const color = d3.scaleOrdinal(d3.schemeCategory10);

            svg.selectAll("myPath")
                .data(trainingData)
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
                .datum(currentData)
                .attr("d", (d) => d3.line()(dimensions.map((p) => [x(p), y[p](d[p])])))
                .style("fill", "none")
                .style("stroke", "green")
                .style("stroke-width", "2px");

            // Add the axes
            for (const dim of dimensions) {
                svg.append("g")
                    .attr("transform", `translate(${x(dim)}, 0)`)
                    .call(d3.axisLeft(y[dim]))
                    .append("text")
                    .attr("y", margin.left * -1)
                    .style("text-anchor", "middle")
                    .attr("y", -9)
                    .text(dim)
                    .style("fill", "black");
            }

            // Assuming 'color' is your d3.scaleOrdinal used for the training data
            const uniqueLabels = color.domain();
            const currDataLabel = 3;
            const currDataColor = "green";

            // Extend the color domain to include the current data
            const extendedColorDomain = uniqueLabels.concat([currDataLabel]);
            const extendedColorRange = color.range().concat([currDataColor]);
            const extendedColorScale = d3
                .scaleOrdinal()
                .domain(extendedColorDomain)
                .range(extendedColorRange);

            // Create a legend group
            const legend = svg
                .append("g")
                .attr("class", "legend")
                .attr("transform", `translate(0, ${height + 20})`)
                .selectAll("g")
                .data(extendedColorDomain)
                .enter()
                .append("g")
                .attr("transform", (d, i) => `translate(${i * 100}, 0)`);

            // Draw rectangles for each legend item
            legend
                .append("rect")
                .attr("width", 10)
                .attr("height", 10)
                .attr("fill", (d) => (d === 3 ? currDataColor : extendedColorScale(d)));

            // Add text labels for each legend item
            legend
                .append("text")
                .attr("x", 15)
                .attr("y", 5)
                .attr("dy", ".35em")
                .text((d) => (d === 1 ? "Phishing" : d === 0 ? "Legitimate" : "Current Data"));
        }
    }, [trainingData, currentData]);

    return (
        <div className="w-full overflow-x-scroll">
            <div ref={d3Container}></div>
        </div>
    );
};

export default ParallelCoordinatesChart;
