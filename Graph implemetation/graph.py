from collections import deque
import heapq
 
class Node:
    def __init__(self, data):
        self.data = data

    # def __str__(self):
    #     return self.data

n1 = Node("Betsegaw")
print(n1)

num = '45'
print(int(num))
    


# class Edge:
#     def __init__(self, start: Node, end: Node, weight: int = 1):
#         self.start = start
#         self.end = end
#         self.weight = weight

#     def __str__(self) -> str:
#         return f'Edge from {self.start.data} to {self.end.data}'
    


class Graph:
    def __init__(self):
        self.adjacencyDic = {}

    def possiblePaths(self, start, end, path=[]):
        path = path + [start]

        if start == end:
            return [path]
        
        if start not in self.adjacencyDic:
            return [] 
        
        paths = []

        for node in self.adjacencyDic[start]:
            if node not in path:
                new_paths = self.possiblePaths(node, end, path)
                for pth in new_paths:
                    paths.append(pth)
        return paths
    
    def createNode(self, data):
        newNode = Node(data)
        self.adjacencyDic[newNode] = []

    def insertEdge(self, start, end, weight):
        if start not in self.adjacencyDic:
            self.adjacencyDic[start] = []
        
        if end not in self.adjacencyDic:
            self.adjacencyDic[end] = []

        self.adjacencyDic[start].append((end, weight))
        self.adjacencyDic[end].append((start, weight))

    
    def deleteEdge(self, start, end, weight):
        self.adjacencyDic[start].remove((end, weight))
        self.adjacencyDic[end].remove((start, weight))

    def deleteNode(self, node_del):
        for neighbour in self.adjacencyDic[node_del]:
            nbr, cost = neighbour
            self.adjacencyDic[nbr].remove(node_del, cost)
        self.adjacencyDic.pop(node_del)
    
    def shortestPath(self, start, end):
        paths = self.possiblePaths(start, end)
        small = [paths[0]]
        for i in paths:
            if len(small[0]) > len(i):
                small = [i]
            elif len(small[0]) == len(i):
                small.append(i)
        return small
    

    def BFS(self, start, target):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            current_node = queue.popleft()
            print(current_node, end=' ')

            for neighbor in self.adjacencyDic.get(current_node, []):
                if neighbor not in visited:
                    if neighbor == target:
                        visited.add(target)
                        return visited
                    queue.append(neighbor)
                    visited.add(neighbor)

    def DFS(self, start, target):
        visited = set()

        def dfs_recursive(node):
            print(node, end=' ')
            visited.add(node)
            for neighbor in self.adjacencyDic.get(node, []):
                if neighbor not in visited:
                    if neighbor == target:
                        visited.add(target)
                        return visited
                    dfs_recursive(neighbor)

        dfs_recursive(start)

    def uniform_cost_search(self, start, end):
        visited = set()
        heap = [(0, start, [])]  # Priority queue: (cost, node, path)

        while heap:
            cost, current_node, path = heapq.heappop(heap)

            if current_node == end:
                return path + [current_node]

            if current_node not in visited:
                visited.add(current_node)

                for edge in self.adjacencyDic.get(current_node.data, []):
                    next_node = edge.end
                    if next_node not in visited:
                        heapq.heappush(heap, (cost + edge.weight, next_node, path + [current_node]))

    def iterative_deepening_search(self, start, end, max_depth=10):
        for depth in range(1, max_depth + 1):
            result = self.depth_limited_search(start, end, depth)
            if result:
                return result

    def depth_limited_search(self, start, end, depth):
        visited = set()

        def dfs(node, path, current_depth):
            if node == end:
                return path + [node]
            if current_depth <= 0:
                return None

            visited.add(node)
            for edge in self.adjacencyDic.get(node.data, []):
                next_node = edge.end
                if next_node not in visited:
                    new_path = dfs(next_node, path + [node], current_depth - 1)
                    if new_path:
                        return new_path
            return None

        return dfs(start, [], depth)

    def bidirectional_search(self, start, end):
        forward_visited = set()
        backward_visited = set()
        forward_queue = deque([(start, [])])
        backward_queue = deque([(end, [])])

        while forward_queue and backward_queue:
            forward_node, forward_path = forward_queue.popleft()
            backward_node, backward_path = backward_queue.popleft()

            if forward_node in backward_visited:
                return forward_path + [forward_node] + backward_path[::-1]

            if backward_node in forward_visited:
                return forward_path[::-1] + [backward_node] + backward_path

            forward_visited.add(forward_node)
            backward_visited.add(backward_node)

            for edge in self.adjacencyDic.get(forward_node.data, []):
                next_node = edge.end
                if next_node not in forward_visited:
                    forward_queue.append((next_node, forward_path + [forward_node]))

            for edge in self.adjacencyDic.get(backward_node.data, []):
                next_node = edge.end
                if next_node not in backward_visited:
                    backward_queue.append((next_node, backward_path + [backward_node]))

    def astar_search(self, start, end, heuristic):
        open_set = [(0 + heuristic(start, end), 0, start, [])]  # (f-cost, g-cost, node, path)

        while open_set:
            _, cost, current_node, path = heapq.heappop(open_set)

            if current_node == end:
                return path + [current_node]

            for edge in self.adjacencyDic.get(current_node.data, []):
                next_node = edge.end
                new_cost = cost + edge.weight
                new_path = path + [current_node]
                f_cost = new_cost + heuristic(next_node, end)
                heapq.heappush(open_set, (f_cost, new_cost, next_node, new_path))

    def shortest_path(self, start, end):
        return self.uniform_cost_search(start, end)


        

