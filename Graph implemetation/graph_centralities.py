from graph import Graph
import numpy as np
from collections import deque
import math
# import networkx as nx

''' 
Compute the Degree, 
Closeness, Eigenvector, Katz, PageRank, and Betweenness centralities on the graph from Question 2.


'''
class GraphCentralities:    
    def adj_list_to_matrix(self, adj_list):
        nodes = list(adj_list.keys())
        n = len(nodes)
        adj_matrix = [[0 for i in range(n)] for j in range(n)]
        
        for node in adj_list:
            row_index = nodes.index(node)
            # print("nodes values", adj_list[node])
            for neighbor in adj_list[node][1]:
                col_index = nodes.index(neighbor[0])
                adj_matrix[row_index][col_index] = 1
                
        return adj_matrix, nodes
    
    def degree_centrality(self, graph):
        degree_centrality = {}
        
        # The degree of a node is the connections it has with other nodes. Which can be calculated easily from an adjacency list.
        for node in graph:
            degree = len(graph[node][1])

            # Calculate the degree centrality for the current node
            centrality = degree / (len(graph) - 1) if len(graph) > 1 else 0
            degree_centrality[node] = centrality

        maximum = max(degree_centrality.values())
        top_ranked = []
        for node, centrality in degree_centrality.items():
            if centrality == maximum: 
                top_ranked.append(graph[node][0].data)
        print("top ranked nodes by degree centrality: ", top_ranked)
        return degree_centrality
    
    
    
        # Calculate the shortest path distances between nodes using BFS
    
    # Compute the closeness centrality for each node
    def closeness_centrality(self, graph):
        def bfs_distances(graph, source):
            # Initialize the distances dictionary with infinite distances for all nodes
            distances = {node: float('inf') for node in graph}
            distances[source] = 0 
            # Use BFS to traverse the graph and update distances
            queue = [source]
            while queue:
                curr_node = queue.pop(0)
                for neighbor,cost in graph[curr_node][1]:
                    total_dist= distances[curr_node] + cost
                    if distances[neighbor] == float('inf'):
                        distances[neighbor]=total_dist
                        queue.append(neighbor)
                    if distances[neighbor]>total_dist:
                        distances[neighbor]=total_dist
            
            return distances
        node_names=[node for node in graph]
        closeness_scores = [0 for i in range(len(node_names))]
        
        close=0
        for i in range(len(node_names)):
            distances = bfs_distances(graph, node_names[i])
            sum_distances = sum(distances.values())
            closeness_scores[i] = (len(node_names) - 1) / sum_distances
            if closeness_scores[i]>close:
                close=closeness_scores[i]
        top_ranked = []
        maximum = max(closeness_scores)
        for i in range(len(closeness_scores)):
            if closeness_scores[i] == maximum:
                top_ranked.append((node_names[i], closeness_scores[i]))
                
        print('top ranked nodes by closeness centrality: ', top_ranked)
        return closeness_scores

    
    

    def eigenvector_centrality(self, graph, max_iter = 100, tol=1e-6):
        
        # Initialize centrality scores with equal weights
        adjacency_matrix, nodes = self.adj_list_to_matrix(graph)
        centrality = np.ones(len(adjacency_matrix))
        centrality /= np.linalg.norm(centrality)

        for _ in range(max_iter):
            new_centrality = np.dot(adjacency_matrix, centrality)

            # Normalizing the centrality scores
            new_centrality /= np.linalg.norm(new_centrality)

            # Checking for convergence
            if np.linalg.norm(new_centrality - centrality, 2) < tol:
                break

            centrality = new_centrality

        maximum = max(centrality)
        top_ranked = []
        
        for i in range(len(centrality)):
            if centrality[i] == maximum:
                top_ranked.append(nodes[i])
        print("top ranked by eigenvector centrality: ", top_ranked)
        return centrality

    def katz_centrality(self, graph, alpha=0.1, beta=1.0, max_iter=100, tol=1e-6):
        adjacency_matrix, nodes = self.adj_list_to_matrix(graph)
        n = len(adjacency_matrix)
        
        # Initialize centrality scores
        centrality = np.zeros(n)
        beta = np.full(n, beta)

        for _ in range(max_iter):
            # Update centrality scores using the Katz centrality equation
            new_centrality = alpha * np.dot(adjacency_matrix, centrality) + beta

            # Check for convergence
            if np.linalg.norm(new_centrality - centrality, 2) < tol:
                break

            centrality = new_centrality

        centrality /= np.linalg.norm(centrality)
        top_ranked = []
        maximum = max(centrality)
        for i in range(len(centrality)):
            if centrality[i] == maximum:
                top_ranked.append((nodes[i], centrality[i]))
                
        print('top ranked by katz centrality: ', top_ranked)
        return centrality
        

    def pagerank(self, graph, d=0.85, max_iter=100, tolerance=1e-6):
        adjacency_matrix, nodes = self.adj_list_to_matrix(graph)
        # Convert the adjacency matrix to a NumPy array
        adjacency_matrix = np.array(adjacency_matrix)

        # Get the number of nodes in the graph
        N = len(adjacency_matrix)

        # Initialize PageRank scores with equal weights
        pagerank_scores = np.ones(N) / N

        for _ in range(max_iter):
            # Normalize the adjacency matrix to represent transition probabilities
            row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.where(np.logical_and(row_sums != 0, ~np.isnan(row_sums), ~np.isnan(adjacency_matrix)), adjacency_matrix / row_sums, 1 / N)

            # Calculate the next iteration of PageRank scores
            new_pagerank_scores = (1 - d) / N + d * np.dot(transition_matrix.T, pagerank_scores)

            # Check for convergence
            if np.linalg.norm(new_pagerank_scores - pagerank_scores, 2) < tolerance:
                break

            pagerank_scores = new_pagerank_scores
            
        top_ranked = []
        maximum = max(pagerank_scores)
        for i in range(len(pagerank_scores)):
            if pagerank_scores[i] == maximum:
                top_ranked.append((nodes[i], pagerank_scores[i]))
                
        print('top ranked by pagerank: ', top_ranked, pagerank_scores)

        return pagerank_scores
    
    def betweenness_centrality(self, graph):
        # initialize variables
        betweenness = {node: 0.0 for node in graph}
        
        # loop over all nodes
        for s in graph:
            queue = deque()
            stack = []
            dist = {node: -1 for node in graph}
            paths = {node: [] for node in graph}
            sigma = {node: 0 for node in graph}
            
            dist[s] = 0
            sigma[s] = 1
            queue.append(s)
            
            while queue:
                v = queue.popleft()
                stack.append(v)
                
                for w, node in graph[v][1]:
                    if dist[w] < 0:
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        paths[w].append(v)
            
            delta = {node: 0 for node in graph}
            while stack:
                w = stack.pop()
                for v in paths[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
                    
        max_between=max(betweenness.values())
        top_ranked = []
        for i in betweenness:
            if betweenness[i]==max_between:
                top_ranked.append((i,max_between))
                
        print('top ranked by betweenness centrality: ', top_ranked)
        return betweenness
            


