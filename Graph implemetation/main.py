from graph import Graph
import random
import timeit
import matplotlib.pyplot as plt

## Question 1
### Romania Map
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

x = romania.astar_search("Arad", "Bucharest", romania.haversine_distance)
y = romania.UCS("Arad", "Bucharest")
w = romania.BFS("Arad", "Bucharest")
z = romania.DFS("Arad", "Bucharest")
a = romania.iterative_deepening_search("Arad", "Bucharest")
b = romania.bidirectional_search("Arad", "Bucharest")
c = romania.greedy("Arad", "Bucharest", romania.haversine_distance)

print(z, romania.find_path_length(z))
print(w, romania.find_path_length(w))
print(y, romania.find_path_length(y))
print(b, romania.find_path_length(b))
print(c, romania.find_path_length(c))
print(a, romania.find_path_length(a))
print(x, romania.find_path_length(x))

### Romaina Map


#------------------------------###------------------------------#


## Question 2
### Random graphs

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
graph_times = []
for g in range(len(Graphs)):
    graph = Graphs[g]
    random_nodes = []
    while len(random_nodes) < 10:
        x = random.randint(1, 10*(g//4+1))
        if x not in random_nodes:
            random_nodes.append(x)

    algos = ["bfs", "dfs", "greedy", "ucs", "iterative_deepening_search", "astar", "bidirectional"]
    
    graph_times.append([])
    graph_paths.append([])
    for algo in algos:
        algo_time = 0   
        algo_path = 0 
        for i in range(len(random_nodes)):
            for j in range(i, len(random_nodes)):
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
                
                    if path:
                        algo_path += graph.find_path_length(path)
                    algo_time += end_time - start_time

        graph_paths[-1].append(algo_path)
        graph_times[-1].append(algo_time/5)


def draw_plot(data, title, y_label):
    # Labels for each index position
    labels = ["bfs", "dfs", "greedy", "ucs", "iterative_deepening_search", "astar", "bidirectional"]

    # Create a list of x values (from 1 to 16)
    x_values = list(range(1, 17))

    # Create a color map with a unique color for each index position
    color_map = plt.get_cmap('Set1')
    num_colors = len(data[0])
    colors = [color_map(i/num_colors) for i in range(num_colors)]

    # Create a figure and axis object
    fig, ax = plt.subplots()
    # Loop through each list in the data and plot as a scatter plot with unique color
    for i in range(len(data)):
        ax.scatter([x_values[i]]*7, data[i], marker="o", color=colors, label=i)

    # Set the x-axis tick labels
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)

    # Set the x and y axis labels
    ax.set_xlabel('Graphs')
    ax.set_ylabel(y_label)

    # Set the plot title
    ax.set_title(title)

    # Create a legend with the labels
    handles, _ = ax.get_legend_handles_labels()

    handles = [plt.Line2D([], [], marker='o', color=colors[i], label=labels[i]) for i in range(len(labels))]
    # labels_legend = [labels[i] for i in range(len(data[0]))]
    ax.legend(handles=handles)

    # Show the plot
    plt.show()

draw_plot(graph_times, "Time analysis", "time taken")
draw_plot(graph_paths, "Path analysis", "Path length")
