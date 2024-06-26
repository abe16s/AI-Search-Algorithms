from collections import deque
import heapq
import random
import math
 
class Node:
    def __init__(self, data, location):
        self.data = data
        self.location = location

    def __str__(self):
        return self.data

class Graph:
    def __init__(self):
        self.adjacencyDic = {}
    
    def createNode(self, data, location):
        newNode = Node(data, location)
        self.adjacencyDic[data] = [newNode, []]

    def insertEdge(self, start, end, weight):
        if start not in self.adjacencyDic:
            self.createNode(start, (random.uniform(1.0, 50) , random.uniform(1.0, 50)))
        
        if end not in self.adjacencyDic:
            self.createNode(end, (random.uniform(1.0, 50) , random.uniform(1.0, 50)))

        self.adjacencyDic[start][1].append((end, weight))
        self.adjacencyDic[end][1].append((start, weight))

    
    def deleteEdge(self, start, end, weight):
        self.adjacencyDic[start][1].remove((end, weight))
        self.adjacencyDic[end][1].remove((start, weight))

    def deleteNode(self, node_del):
        for neighbor in self.adjacencyDic[node_del]:
            nbr, cost = neighbor
            self.adjacencyDic[nbr].remove(node_del, cost)
        self.adjacencyDic.pop(node_del)
    

    # Searches 
    def BFS(self, start, target):
        queue = deque([start])
        parent = {start: start}

        while queue:
            current_node = queue.popleft()
            if current_node == target:
                path = []
                while current_node != start:
                    path.append(current_node)
                    current_node = parent[current_node]
                
                path.append(start)
                return path[::-1]
            
            for neighbor, cost in self.adjacencyDic[current_node][1]:
                if neighbor not in parent:
                    queue.append(neighbor)
                    parent[neighbor] = current_node

        return None


    def DFS(self, start, target):
        stack = [start]
        parent = {start: None}

        while stack:
            current_node = stack.pop()
            if current_node == target:
                path = []
                while current_node != start:
                    path.append(current_node)
                    current_node = parent[current_node]
                
                path.append(start)
                return path[::-1]

            for neighbor, cost in self.adjacencyDic[current_node][1]:
                if neighbor not in parent:
                    stack.append(neighbor)
                    parent[neighbor] = current_node

        return None

    def UCS(self, start, end):
        visited = set()
        heap = [(0, start, [])]  # Priority queue: (cost, node, path)

        while heap:
            cost, current_node, path = heapq.heappop(heap)

            if current_node == end:
                return path + [current_node]

            visited.add(current_node)

            for neighbor, cur_cost in self.adjacencyDic[current_node][1]:
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + cur_cost, neighbor, path + [current_node]))

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
            for neighbor, cost in self.adjacencyDic[node][1]:
                if neighbor not in visited:
                    new_path = dfs(neighbor, path + [node], current_depth - 1)
                    if new_path:
                        return new_path
            return None

        return dfs(start, [], depth)

    def bidirectional_search(self, start, end):
        forward_parent = {start: None}
        backward_parent = {end: None}
        forward_queue = deque([start])
        backward_queue = deque([end])

        while forward_queue and backward_queue:
            forward_node = forward_queue.popleft()
            backward_node = backward_queue.popleft()

            if forward_node in backward_parent or backward_node in forward_parent:
                cur = forward_node if forward_node in backward_parent else backward_node
                path = []
                forw = cur
                while forw != start:
                    path.append(forw)
                    forw = forward_parent[forw]
                path.append(start)
                path.reverse()

                while cur != end:
                    cur = backward_parent[cur]
                    path.append(cur)
                return path

            for neighbor, cost in self.adjacencyDic[forward_node][1]:
                if neighbor not in forward_parent:
                    forward_queue.append(neighbor)
                    forward_parent[neighbor] = forward_node

            for neighbor, cost in self.adjacencyDic[backward_node][1]:
                if neighbor not in backward_parent:
                    backward_queue.append(neighbor)
                    backward_parent[neighbor] = backward_node
    
    
    def astar_search(self, start, end, heuristic):
        open_set = [(0 + heuristic(start, end), 0, start, [])]  # (f-cost, g-cost, node, path)
        visited = set()
        while open_set:
            _, cost, current_node, path = heapq.heappop(open_set)
            if current_node == end:
                return path + [current_node]
            
            visited.add(current_node)

            for neighbor, cur_cost in self.adjacencyDic[current_node][1]:
                if neighbor not in visited:
                    new_cost = cost + cur_cost
                    new_path = path + [current_node]
                    f_cost = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_cost, new_cost, neighbor, new_path))

    def greedy(self, start, end, heuristic):
        current = (heuristic(start, end), start, [])  # (heuristic, node, path)
        visited = set()
        while current:
            heu_cost, current_node, path = current
            visited.add(current_node)
            current = None
            mini = float("inf")
            if current_node == end:
                return path + [current_node]

            for neighbor, cur_cost in self.adjacencyDic[current_node][1]:
                if neighbor not in visited:
                    new_path = path + [current_node]
                    cur = heuristic(neighbor, end)
                    if cur < mini:
                        current = (cur, neighbor, new_path)
                        mini = cur

    def haversine_distance(self, node_a, node_b):

        lat1, lon1 = self.adjacencyDic[node_a][0].location
        lat2, lon2 = self.adjacencyDic[node_b][0].location

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        radius_earth = 6371  
        distance = radius_earth * c

        return distance

    def find_path_length(self, path):
        total_cost = 0
        for p in range(len(path)-1):
            cur = path[p]
            for nbr, cost in self.adjacencyDic[cur][1]:
                if nbr == path[p+1]:
                    total_cost += cost
        return total_cost

            

