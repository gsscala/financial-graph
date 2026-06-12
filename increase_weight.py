import networkx as nx

input_file = "negative1_graph.gexf"
output_file = "visualize_negative1_graph.gexf"

# Load graph
G = nx.read_gexf(input_file)

# Add 1.1 to every edge weight
for u, v, data in G.edges(data=True):
    old_weight = data.get("weight", 0)
    data["weight"] = float(old_weight) + 1.1

# Save rewritten GEXF
nx.write_gexf(G, output_file)

print(f"Updated graph saved to {output_file}")