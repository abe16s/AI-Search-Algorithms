from graph import Graph
import numpy as np

''' 
Compute the Degree, 
Closeness, Eigenvector, Katz, PageRank, and Betweenness centralities on the graph from Question 2.


'''
class GraphCentralities:    
    def adj_list_to_matrix(self, adj_list):
        nodes = list(adj_list.keys())
        print(nodes)
        n = len(nodes)
        adj_matrix = [[0 for i in range(n)] for j in range(n)]
        
        for node in adj_list:
            row_index = nodes.index(node)
            # print("nodes values", adj_list[node])
            for neighbor in adj_list[node][1]:
                col_index = nodes.index(neighbor[0])
                adj_matrix[row_index][col_index] = 1
                
        return adj_matrix
    
    def degree_centrality(self, graph):
        degree_centrality = {}
        
        # The degree of a node is the connections it has with other nodes. Which can be calculated easily from an adjacency list.
        for node in graph:
            degree = len(graph[node])

            # Calculate the degree centrality for the current node
            centrality = degree / (len(graph) - 1) if len(graph) > 1 else 0
            degree_centrality[node] = centrality

        return degree_centrality
    
    
    def closeness_centrality(self, graph):
        def calculate_shortest_lengths(node):
            # Perform breadth-first search to compute shortest path lengths
            shortest_lengths = {node: 0}
            queue = [node]

            while queue:
                current_node = queue.pop(0)
                for neighbor in graph[current_node][1]:
                    if neighbor[0] not in shortest_lengths:
                        shortest_lengths[neighbor[0]] = shortest_lengths[current_node] + 1
                        queue.append(neighbor[0])

            return shortest_lengths
        def node_closeness_centrality(node):
            shortest_lengths = calculate_shortest_lengths(node)
            total_shortest_lengths = sum(shortest_lengths.values())

            # Calculate the closeness centrality
            if len(graph) > 1 and total_shortest_lengths:
                closeness_centrality = (len(shortest_lengths) - 1) / total_shortest_lengths
            else:
                closeness_centrality = 0

            return closeness_centrality
        
        closeness_centralities = {}
        for node in graph:
            closeness_centralities[node] = node_closeness_centrality(node)
            
        return closeness_centralities


    def eigenvector_centrality(self, graph, max_iter = 100, tol=1e-6):
        
        # Initialize centrality scores with equal weights
        adjacency_matrix = self.adj_list_to_matrix(graph)
        centrality = np.ones(len(adjacency_matrix))

        for _ in range(max_iter):
            new_centrality = np.dot(adjacency_matrix, centrality)

            # Normalizing the centrality scores
            new_centrality /= np.linalg.norm(new_centrality, 1)

            # Checking for convergence
            if np.linalg.norm(new_centrality - centrality, 2) < tol:
                break

            centrality = new_centrality

        print(centrality)
        return centrality

    def katz_centrality(self, graph, alpha=0.1, beta=1.0, max_iter=100, tol=1e-6):
        adjacency_matrix = self.adj_list_to_matrix(graph)
        
        n = len(adjacency_matrix)
        
        
        # Initialize centrality scores
        centrality = np.ones(n)

        for _ in range(max_iter):
            # Update centrality scores using the Katz centrality equation
            new_centrality = alpha * np.dot(adjacency_matrix, centrality) + beta

            # Check for convergence
            if np.linalg.norm(new_centrality - centrality, 2) < tol:
                break

            centrality = new_centrality

        return centrality
        

    def pagerank(self, graph, d=0.85, max_iter=100, tolerance=1e-6):
        adjacency_matrix = self.adj_list_to_matrix(graph)
        # Convert the adjacency matrix to a NumPy array
        adjacency_matrix = np.array(adjacency_matrix)

        # Get the number of nodes in the graph
        N = len(adjacency_matrix)

        # Initialize PageRank scores with equal weights
        pagerank_scores = np.ones(N) / N

        for _ in range(max_iter):
            # Normalize the adjacency matrix to represent transition probabilities
            row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.where(row_sums != 0, adjacency_matrix / row_sums, 1 / N)

            # Calculate the next iteration of PageRank scores
            new_pagerank_scores = (1 - d) / N + d * np.dot(transition_matrix.T, pagerank_scores)

            # Check for convergence
            if np.linalg.norm(new_pagerank_scores - pagerank_scores, 2) < tolerance:
                break

            pagerank_scores = new_pagerank_scores

        return pagerank_scores

    def betweenness_centrality(self, graph):
        # Initialize dictionary to store betweenness centrality for each node
        betweenness_centrality = {node: 0.0 for node in graph}

        # Iterate over all nodes as potential sources
        for source in graph:
            # Initialize dictionary to store shortest paths count and predecessor for each node
            shortest_paths_count = {node: 0 for node in graph}
            predecessors = {node: [] for node in graph}
            shortest_paths_count[source] = 1

            # Breadth-first search to find shortest paths and count them
            queue = [source]
            while queue:
                current_node = queue.pop(0)
                
                for neighbor in graph[current_node][1]:
                    if shortest_paths_count[neighbor[0]] == 0:
                        queue.append(neighbor[0])
                        shortest_paths_count[neighbor[0]] = shortest_paths_count[current_node]
                        predecessors[neighbor[0]].append(current_node)
                    elif shortest_paths_count[neighbor[0]] == shortest_paths_count[current_node] + 1:
                        predecessors[neighbor[0]].append(current_node)

            # Calculate dependencies and accumulate betweenness centrality for each node
            dependencies = {node: 0.0 for node in graph}
            while predecessors:
                node = max(predecessors, key=lambda x: shortest_paths_count[x])
                for predecessor in predecessors[node]:
                    dependencies[predecessor] += (1 + dependencies[node]) / len(predecessors[node])
                if node != source:
                    betweenness_centrality[node] += dependencies[node]
                predecessors.pop(node)

        # Normalize betweenness centrality values
        n = len(graph) - 1
        for node in betweenness_centrality:
            betweenness_centrality[node] /= (n * (n + 1))

        return betweenness_centrality

        
        
        