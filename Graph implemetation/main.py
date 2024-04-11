from graph import Graph
import random
import timeit
from collections import defaultdict



def generate_random_graph(n, p):
    # graph = Graph()
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j and random.random() < p:
                # graph.insertEdge(i, j, 1)
                pass
    # return graph


graph1 = generate_random_graph(10, 0.2)
graph2 = generate_random_graph(10, 0.4)
graph3 = generate_random_graph(10, 0.6)
graph4 = generate_random_graph(10, 0.8)

graph5 = generate_random_graph(20, 0.2)
graph6 = generate_random_graph(20, 0.4)
graph7 = generate_random_graph(20, 0.6)
graph8 = generate_random_graph(20, 0.8)

graph9 = generate_random_graph(30, 0.2)
graph10 = generate_random_graph(30, 0.4)
graph11 = generate_random_graph(30, 0.6)
graph12 = generate_random_graph(30, 0.8)

graph13 = generate_random_graph(40, 0.2)
graph14 = generate_random_graph(40, 0.4)
graph15 = generate_random_graph(40, 0.6)
graph16 = generate_random_graph(40, 0.8)


Graphs = [graph1, graph2, graph3, graph4, graph5, graph6, graph7, graph8, graph9, graph10, graph11, graph12, graph13, graph14, graph15, graph16]
# graph_paths = []
# graph_times = []
for g in range(len(Graphs)):
    graph = Graphs[g]
    random_nodes = []
    while len(random_nodes) < 10:
        x = random.randint(1, 10*(g//4+1))
        if x not in random_nodes:
            random_nodes.append(x)

    # print(random_nodes)
    algos = ["dfs", "bfs", "ucs", "bidirectional", "greedy", "astar", "iterative_deepening_search"]
    algo_path_length = defaultdict(list)
    algo_time_taken = defaultdict(int)
    
    # for i in range(len(random_nodes)):
    #     for j in range(i, len(random_nodes)):
    #         for algo in algos:
    #             algo_path = 0
    #             algo_time = 0    
    #             for _ in range(5):
    #                 start_time = timeit.default_timer()
    #                 if algo == "dfs":
    #                     path = dfs(random_nodes[i], random_nodes[j], graph.graph)
    #                 elif algo == "bfs":
    #                     path = bfs(random_nodes[i], random_nodes[j], graph.graph)
    #                 elif algo == "greedy":
    #                     path = greedy(graph.graph, random_nodes[i], random_nodes[j], graph.locations)
    #                 elif algo == "iterative_deepening_search":
    #                     path = iterative_deepening_search(graph.graph, random_nodes[i], random_nodes[j], 4)
    #                 elif algo == "astar":
    #                     path = astar(graph.graph, random_nodes[i], random_nodes[j], graph.locations)
    #                 elif algo == "ucs":
    #                     path = ucs(graph, random_nodes[i], random_nodes[j])
    #                 elif algo == "bidirectional":
    #                     path = bidirectional(graph, random_nodes[i], random_nodes[j])

    #                 end_time = timeit.default_timer()
                
    #             if path:
    #                 algo_path += find_taken_distance(path, graph.graph)
    #             algo_time += end_time - start_time

    #         algo_path_length[algo].append(algo_path/5)
    #         algo_time_taken[algo].append(algo_time/5)

    # graph_paths.append(algo_path_length)
    # graph_times.append(algo_time_taken)

cities = {}
with open('Graph implemetation\cities.txt', 'r') as file:
    for line in file:
        c = line.strip().split()
        if c[0] == "City":
            continue
        cities[c[0]] = (c[1], c[2])
