import networkx as nx
import json
import os
import webbrowser
import threading
import socketserver
import http.server

# Load GraphML file and transfer to JSON
def graphml_to_json(graphml_file):
    G = nx.read_graphml(graphml_file)
    data = nx.node_link_data(G)
    return json.dumps(data)

# Create HTML file with improved visualization
def create_html(html_path):
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: Arial, sans-serif;
        }
        svg {
            width: 100%;
            height: 100%;
            background: #f0f0f0;
        }
        .links line {
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }
        .nodes circle {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node-label {
            font-size: 14px;
            pointer-events: none;
            fill: #333;
        }
        .link-label {
            font-size: 12px;
            fill: #666;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .link:hover .link-label {
            opacity: 1;
        }
        .tooltip {
            position: absolute;
            text-align: left;
            padding: 10px;
            font: 14px sans-serif;
            background: lightsteelblue;
            border: 1px solid #ccc;
            border-radius: 8px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 300px;
        }
        .legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        .legend-item {
            margin: 5px 0;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <svg></svg>
    <div class="tooltip"></div>
    <div class="legend"></div>
    <script type="text/javascript" src="./graph_json.js"></script>
    <script>
        const graphData = graphJson;

        const svg = d3.select("svg"),
            width = window.innerWidth,
            height = window.innerHeight;

        svg.attr("viewBox", [0, 0, width, height]);

        const g = svg.append("g");

        // Extract entity types and assign colors
        const entityTypes = [...new Set(graphData.nodes.map(d => d.entity_type))];
        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(entityTypes);

        // Force simulation setup
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-500))  // Increased repulsion
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(30));

        // Links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("stroke-width", d => Math.sqrt(d.value || 1))
            .attr("stroke", "#999");

        // Link labels
        const linkLabel = g.append("g")
            .attr("class", "link-labels")
            .selectAll("text")
            .data(graphData.links)
            .enter().append("text")
            .attr("class", "link-label")
            .text(d => d.description || "")
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle");

        // Nodes
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(graphData.nodes)
            .enter().append("circle")
            .attr("r", 10)  // Increased node size
            .attr("fill", d => color(d.entity_type))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Node labels
        const nodeLabel = g.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(graphData.nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.id)
            .attr("x", d => d.x + 15)
            .attr("y", d => d.y + 5);

        // Tooltip
        const tooltip = d3.select(".tooltip");

        node.on("mouseover", function(event, d) {
            tooltip.transition()
                .duration(200)
                .style("opacity", 0.9);
            tooltip.html(`<strong>${d.id}</strong><br>Entity Type: ${d.entity_type}<br>Description: ${d.description || "N/A"}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

        // Legend
        const legend = d3.select(".legend");
        entityTypes.forEach(type => {
            legend.append("div")
                .attr("class", "legend-item")
                .html(`<span class="legend-color" style="background-color: ${color(type)}"></span>${type}`);
        });

        // Simulation tick
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            nodeLabel
                .attr("x", d => d.x + 15)
                .attr("y", d => d.y + 5);
        });

        // Drag functions
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }

        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }

        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", zoomed);

        svg.call(zoom);

        function zoomed(event) {
            g.attr("transform", event.transform);
        }
    </script>
</body>
</html>
    '''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# Create JSON file
def create_json(json_data, json_path):
    json_data = "var graphJson = " + json_data.replace('\\"', '').replace("'", "\\'").replace("\n", "")
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(json_data)

# Start simple HTTP server
def start_server(port):
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Server started at http://localhost:{port}")
        httpd.serve_forever()

# Main function
def visualize_graphml(graphml_file, html_path, port=8000):
    json_data = graphml_to_json(graphml_file)
    html_dir = os.path.dirname(html_path)
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    json_path = os.path.join(html_dir, 'graph_json.js')
    create_json(json_data, json_path)
    create_html(html_path)
    # Start server in background
    server_thread = threading.Thread(target=start_server, args=(port,))
    server_thread.daemon = True
    server_thread.start()
    # Open default browser
    webbrowser.open(f'http://localhost:{port}/{os.path.basename(html_path)}')
    print("Visualization is ready. Press Ctrl+C to exit.")
    try:
        # Keep main thread running
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")

# Usage
if __name__ == "__main__":
    graphml_file = r"./work_dir_base_small/graph_chunk_entity_relation.graphml"  # Replace with your GraphML file path
    html_path = "./graph_visualization.html"
    visualize_graphml(graphml_file, html_path, 11236)