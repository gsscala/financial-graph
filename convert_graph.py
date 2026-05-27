import networkx as nx
import csv
import numpy as np

G = nx.MultiDiGraph()

with open("graph.csv", "r") as f:
    reader = csv.reader(f)
    for node1, node2, weight, time in reader:
        node1 = int(node1)
        node2 = int(node2)
        weight = (int(weight))/10
        G.add_edge(node1, node2, weight=weight)

nx.write_gexf(G, "graph_jupyter.gexf")