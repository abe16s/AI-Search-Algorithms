from graph import Graph
import random
import timeit
from collections import defaultdict

def generate_random_graph(n, p):
    random_graph = Graph()
    for i in range(1, n + 1):
        if i not in random_graph.adjacencyDic:
            random_graph.createNode(i, (random.uniform(1.0, 50) , random.uniform(1.0, 50)))
        for j in range(i+1, n + 1):
            if random.random() < p:
                random_graph.insertEdge(i, j, random.randint(1,100))
    return random_graph

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

graph_paths = []
graph_times = defaultdict(list)
for g in range(len(Graphs)):
    graph = Graphs[g]
    random_nodes = []
    while len(random_nodes) < 10:
        x = random.randint(1, 10*(g//4+1))
        if x not in random_nodes:
            random_nodes.append(x)

    algos = ["bfs", "dfs", "greedy", "ucs", "iterative_deepening_search", "astar", "bidirectional"]
    algo_path_length = defaultdict(list)
    algo_time_taken = defaultdict(list)
    
    for algo in algos:
        algo_time = 0    
        for i in range(len(random_nodes)):
            for j in range(i, len(random_nodes)):
                # algo_path = 0
                for _ in range(5):
                    start_time = timeit.default_timer()
                    if algo == "dfs":
                        path = graph.DFS(random_nodes[i], random_nodes[j])
                    elif algo == "bfs":
                        path = graph.BFS(random_nodes[i], random_nodes[j])
                    elif algo == "greedy":
                        path = graph.greedy(random_nodes[i], random_nodes[j], graph.haversine_distance)
                    elif algo == "iterative_deepening_search":
                        path = graph.iterative_deepening_search(random_nodes[i], random_nodes[j], 10)
                    elif algo == "astar":
                        path = graph.astar_search(random_nodes[i], random_nodes[j], graph.haversine_distance)
                    elif algo == "ucs":
                        path = graph.UCS(random_nodes[i], random_nodes[j])
                    elif algo == "bidirectional":
                        path = graph.bidirectional_search(random_nodes[i], random_nodes[j])

                    end_time = timeit.default_timer()
                
                    # if path:
                    #     algo_path += find_taken_distance(path, graph.graph)
                    algo_time += end_time - start_time

                # algo_path_length[algo].append(algo_path/5)
                # algo_time_taken[algo] += (algo_time)

        # graph_paths.append(algo_path_length)
        graph_times[algo].append(algo_time)# (algo_time_taken)

for k in graph_times:
    print("\n",k)
    print(graph_times[k])


romania = Graph()
with open('Graph implemetation\cities.txt', 'r') as file:
    for line in file:
        c = line.strip().split()
        if c[0] == "City":
            continue
        romania.createNode(c[0], (float(c[1]), float(c[2])))

romania.insertEdge('Oradea','Sibiu', 151)
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

# print(romania.astar_search("Oradea", "Neamt", romania.haversine_distance))
# print(romania.UCS("Oradea", "Neamt"))
# print(romania.astar_search("Fagaras", "Neamt", romania.haversine_distance))
# print(romania.bidirectional_search("Lugoj", "Urziceni"))
# print(romania.BFS("Lugoj", "Urziceni"))
