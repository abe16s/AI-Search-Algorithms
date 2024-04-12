from graph import Graph
import random
import timeit
from collections import defaultdict
from graph_centralities import GraphCentralities 

romania = Graph()
with open('Graph implemetation\cities.txt', 'r') as file:
    for line in file:
        c = line.strip().split()
        if c[0] == "City":
            continue
        romania.createNode(c[0], (float(c[1]), float(c[2])))

with open('Graph implemetation\edges.txt', 'r') as file:
    for line in file:
        c = line.strip().split()
        romania.insertEdge(c[0], c[1], int(c[2]))
        
        
graphs = [romania]
gc = GraphCentralities()
for graph in graphs:
    print("=====================ANSWER==========================")
    gc.degree_centrality(graph.adjacencyDic)
    gc.closeness_centrality(graph.adjacencyDic)
    gc.eigenvector_centrality(graph.adjacencyDic)
    gc.katz_centrality(graph.adjacencyDic)
    gc.pagerank(graph.adjacencyDic)
    gc.betweenness_centrality(graph.adjacencyDic)
    print("=====================================================")







