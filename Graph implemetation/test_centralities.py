from graph import Graph
import random
import timeit
from collections import defaultdict
from graph_centralities import GraphCentralities 
import networkx as nx

def generate_random_graph(n, p):
    random_graph = Graph()
    for i in range(1, n + 1):
        if i not in random_graph.adjacencyDic:
            random_graph.createNode(i, (random.uniform(1.0, 50) , random.uniform(1.0, 50)))
        for j in range(i+1, n + 1):
            if random.random() < p:
                random_graph.insertEdge(i, j, random.randint(1,100))
    return random_graph


gc = GraphCentralities()
romania = Graph()
romania.insertEdge('Oradea','Sibiu', 151 )
romania.insertEdge('Oradea','Zerind', 71)
romania.insertEdge('Zerind','Arad', 75)
romania.insertEdge('Arad','Sibiu', 140)
romania.insertEdge('Arad','Timisoara', 118)
romania.insertEdge('Sibiu','Fagaras', 99)
romania.insertEdge('Sibiu','Rimnicu_Vilcea', 80)
romania.insertEdge('Timisoara','Lugoj', 111)
romania.insertEdge('Lugoj','Mehadia', 70)
romania.insertEdge('Mehadia','Drobeta', 75)
romania.insertEdge('Drobeta','Craiova', 120)
romania.insertEdge('Craiova','Pitesti', 138)
romania.insertEdge('Rimnicu_Vilcea','Pitesti', 97)
romania.insertEdge('Rimnicu_Vilcea', 'Craiova', 146)
romania.insertEdge('Fagaras', 'Bucharest', 211)
romania.insertEdge('Pitesti', 'Bucharest', 101)
romania.insertEdge('Bucharest', 'Urziceni', 85)
romania.insertEdge('Bucharest', 'Giurgiu', 90)
romania.insertEdge('Urziceni', 'Hirsova', 98)
romania.insertEdge('Urziceni', 'Vaslui', 142)
romania.insertEdge('Hirsova', 'Eforie', 86)
romania.insertEdge('Vaslui', 'Iasi', 92)
romania.insertEdge('Iasi', 'Neamt', 87)
graphs = [romania, generate_random_graph(10, 0.2), generate_random_graph(10, 0.4)]
for graph in graphs:
    print("=====================ANSWER==========================")
    gc.degree_centrality(graph.adjacencyDic)
    gc.closeness_centrality(graph.adjacencyDic)
    gc.eigenvector_centrality(graph.adjacencyDic)
    gc.katz_centrality(graph.adjacencyDic)
    gc.pagerank(graph.adjacencyDic)
    gc.betweenness_centrality(graph.adjacencyDic)
    print("=====================================================")







