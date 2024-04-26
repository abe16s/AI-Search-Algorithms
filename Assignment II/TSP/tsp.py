import argparse
from graph import UndirectedGraph
from hill_climbing import hill_climbing
from simulated_annealing import simulated_annealing
from genetic_algorithm import genetic_algorithm
import utils


parser = argparse.ArgumentParser(
    description="Solve the TSP using a specified algorithm"
)
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["hc", "sa", "ga"],
    default="ga",
    help="the algorithm to use (default: ga)",
)
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="the path to the file containing the TSP data",
)


args = parser.parse_args()
graph = UndirectedGraph()


with open(args.file) as file:
    cities = set()
    for line in file.readlines()[1:]:
        city_1, city_2, distance = line.split("    ")

        distance = int(distance)

        graph.add_edge(city_1, city_2, distance)
        cities.add(city_1)
        cities.add(city_2)

    cities = list(cities)


import time

# Solve the TSP using the specified algorithm
if args.algorithm == "hc":
    tour, dist = hill_climbing(cities=cities, graph=graph, generation=100)

elif args.algorithm == "sa":
    tour, dist = simulated_annealing(cities=cities, graph=graph, generation=10000)

else:
    x = time.perf_counter()
    tour, fitness = genetic_algorithm(
        cities=cities, graph=graph, population_size=150, percent=0.7, generation=200
    )
    print(time.perf_counter() - x)


print("Algorithm:", args.algorithm)
print("Best tour found:", tour)
print("Total cost:", utils.calculate_path_distance(tour, graph))
