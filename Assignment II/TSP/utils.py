import time
import random
from math import radians, cos, sqrt, asin
from graph import *
from heapq import heappop, heappush

def build_path(path_map: dict, start_node, end_node):
    path = []

    current = end_node
    while True:
        path.append(current)

        if current == start_node:
            break

        current = path_map[current]

    path.reverse()

    return path


def uniform_cost_search(start_node, end_node, graph):
    path_map = {}
    path_cost = {}

    priority_queue = [(0, start_node)]

    visited = set()

    while priority_queue:
        # print(path_map)
        cost, current_node = heappop(priority_queue)

        if current_node == end_node:
            return cost, build_path(path_map, start_node, end_node)

        visited.add(current_node)

        for neighbour, weight in graph.neighbours(current_node):
            if neighbour not in visited:
                heappush(priority_queue, (cost + weight, neighbour))

                if path_cost.get(neighbour, float("inf")) > cost + weight:
                    path_map[neighbour] = current_node
                    path_cost[neighbour] = cost + weight

    return float("inf"), []


def build_full_path(chromosome, graph):
    
    total_path = []

    for i in range(1, len(chromosome)):
        city1, city2 = chromosome[i - 1], chromosome[i]

        path_cost, path = uniform_cost_search(city1, city2, graph=graph)

        total_path.extend(path)
        total_path.pop()

    return total_path



def calculate_cost(chromosome, graph):

    cost = 0
    visited = set()


    for i in range(1, len(chromosome)):

        city1, city2 = chromosome[i - 1], chromosome[i]        
        
        path_cost, path = uniform_cost_search(city1, city2, graph=graph)           

        cost += path_cost

        for city in path[1:]:
            if city in visited:
                cost += 1000
        
        visited.update(path)
    
    cost += abs(len(graph.nodes) - len(visited)) * 6000

    return cost

def calculate_path_distance(path, graph):
    cost = 0
    
    for i in range(1, len(path)):

        city1, city2 = path[i - 1], path[i]        
        
        path_cost, _ = uniform_cost_search(city1, city2, graph=graph)           

        cost += path_cost    

    return cost



def benchmark(algorithm, args, run_n_times=10):
    """
    Calculate average runing time for given function

    Args:
        algorithm (Callable): the function to test
        args (Any): the arguments to pass to the function
        run_n_times (int): number of times to run the algorithm

    Return:
        tuple(float, int) : 
            mean_run_time   - average running time of algorithm        
            solution_length - path length
    """

    mean_run_time = 0
    solution_length = 0

    for _ in range(run_n_times):
        start_time = time.perf_counter()
        solution = algorithm(**args)
        mean_run_time += time.perf_counter() - start_time

        solution_length = len(solution)

    return (mean_run_time / run_n_times, solution_length)
