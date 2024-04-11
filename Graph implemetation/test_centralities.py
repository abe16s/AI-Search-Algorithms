from graph import Graph
import random
import timeit
from collections import defaultdict
from graph_centralities import GraphCentralities 

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
graph1 = generate_random_graph(10, 0.2)
graph2 = generate_random_graph(10, 0.4)
graph3 = generate_random_graph(10, 0.6)
graph4 = generate_random_graph(10, 0.8)
g = [graph1, graph2, graph3,graph4]
for gs in g:
    print(gc.degree_centrality(gs.adjacencyDic))
    print(gc.closeness_centrality(gs.adjacencyDic))
    print(gc.eigenvector_centrality(gs.adjacencyDic))
    print(gc.katz_centrality(gs.adjacencyDic))
    print(gc.pagerank(gs.adjacencyDic))
    print(gc.betweenness_centrality(gs.adjacencyDic))
    

# graph5 = generate_random_graph(20, 0.2)
# graph6 = generate_random_graph(20, 0.4)
# graph7 = generate_random_graph(20, 0.6)
# graph8 = generate_random_graph(20, 0.8)

# graph9 = generate_random_graph(30, 0.2)
# graph10 = generate_random_graph(30, 0.4)
# graph11 = generate_random_graph(30, 0.6)
# graph12 = generate_random_graph(30, 0.8)

# graph13 = generate_random_graph(40, 0.2)
# graph14 = generate_random_graph(40, 0.4)
# graph15 = generate_random_graph(40, 0.6)
# graph16 = generate_random_graph(40, 0.8)


# Graphs = [graph1, graph2, graph3, graph4, graph5, graph6, graph7, graph8, graph9, graph10, graph11, graph12, graph13, graph14, graph15, graph16]
