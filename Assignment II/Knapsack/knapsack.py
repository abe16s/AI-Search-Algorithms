import sys
import argparse
import random
import numpy as np
from typing import List


def load_knapsack_problem(filename: str):
    with open(filename) as f:
        lines = f.readlines()

        if len(lines) < 3:
            raise ValueError()

        max_weight = int(lines[0].strip())

        items = {
            "names": [],
            "prices": [],
            "weights": [],
            "counts": [],
        }

        for line_no, line in enumerate(lines[2:]):
            try:
                item_name, weight, value, count = map(
                    lambda string: string.strip(), line.split(",")
                )
                items["names"].append(item_name)
                items["prices"].append(float(value))
                items["weights"].append(float(weight))
                items["counts"].append(int(count))

            except Exception as e:
                raise ValueError(
                    f"Parse error '{filename}' on line {line_no + 3}: '{line}'\n[{e}]"
                )

        for key in items.keys():
            items[key] = np.array(items[key])
    
    return max_weight, items


def calculate_fitness(
    items: np.array, prices: np.array, weights: np.array, max_capacity: float
):
    """Calculates total value of items in samples

    Args:
        items (np.array): array of integers where
                            index indicate the item and
                            value represents item count
        prices (np.array)  : array of integers where
                            index indicate the item and
                            value represents item price
        max_capacity (float): maximum weight capacity of bag

    Returns:
        float: total price of items if their total weight < max_capacity, 0 otherwise
    """

    total_weight = np.sum(items * weights)

    if total_weight > max_capacity:
        return -1

    total_price = np.sum(items * prices)

    return total_price


#########################
""" Genetic Algorithm """
#########################


def generate_population(item_counts: np.array, population_size: int = 50):
    population = []

    for row in np.identity(len(item_counts), dtype=np.int32):
        population.append(row)

    random_population = []

    for item_count in item_counts:
        column = np.random.randint(item_count + 1, size=population_size)

        random_population.append(column)

    for row in np.array(random_population).transpose():
        population.append(row)
    
    return population


def crossover(parent_1: np.array, parent_2: np.array) -> np.array:
    split_index = random.randrange(len(parent_1))

    return np.concatenate((parent_1[:split_index], parent_2[split_index:]))


def mutate(sample: np.array, item_counts: np.array):
    mutation_index = random.randrange(len(sample))

    sample[mutation_index] = random.randrange(item_counts[mutation_index] + 1)


def genetic_algorithm(
    problem,
    initial_population: List[np.array],
    max_capacity: int,
    n_generation: int = 150,
    kept_parents: float = 0.7,
    crossover_mutation: float = 0.1,
    random_mutation: float = 0.05,
):
    """
    Optimize knapsack problem using genetic algorithm

    Args:
        problem (dict): dictionary containing item name, price, weight, total count
        initial_population (np.array): array of initial population
        max_capacity (int): capacity of bag
        n_generation (int): number of iteration genetric algorithm will run
        kept_parents (float): size of top population that will be passed to next generation
        crossover_mutation (float): mutation probability during crossover
        random_mutation (float): mutation probability on random sample

    Return:
        np.array : array of integer where a[i] is number of items we should take of ith item
    """

    prices = problem["prices"]
    weights = problem["weights"]
    counts = problem["counts"]

    population = initial_population

    length = len(population)

    population.sort(
        key=lambda sample: calculate_fitness(sample, prices, weights, max_capacity),
        reverse=True,
    )

    for _ in range(n_generation):
        top_population = population[: int(length * kept_parents)]

        next_generation = []

        for _ in range(length - len(top_population)):
            parent_1, parent_2 = random.choices(top_population, k=2)

            child = crossover(parent_1, parent_2)

            if random.uniform(0, 1) <= crossover_mutation:
                mutate(child, counts)

            if random.uniform(0, 1) <= random_mutation:
                mutate(population[random.randrange(len(population))], counts)

            next_generation.append(child)

        population = top_population + next_generation

        population.sort(
            key=lambda sample: calculate_fitness(sample, prices, weights, max_capacity),
            reverse=True,
        )

    return population[0]



#####################
""" Hill climbing """
#####################


def next_states(state, problem, max_capacity):
    max_count = problem["counts"]
    prices = problem["prices"]
    weights = problem["weights"]

    neighbours = []

    # adding 1 item
    for i in range(len(state)):
        next_state = state[:]

        if next_state[i] < max_count[i]:
            next_state[i] += 1
            neighbours.append(next_state)

    # adding 1 item after removing other item
    for i in range(len(state)):
        if state[i] == 0:
            continue

        next_state = state[:]
        next_state[i] -= 1

        for j in range(i + 1, len(state)):
            if next_state[j] < max_count[i]:
                next_state[j] += 1
                neighbours.append(next_state)

    # swap items
    for idx_1 in range(len(state)):
        for idx_2 in range(idx_1 + 1, len(state)):
            next_state = state[:]
            next_state[idx_1], next_state[idx_2] = next_state[idx_2], next_state[idx_1]
            neighbours.append(next_state)

    # drop items if current state is invalid until it's valid
    if calculate_fitness(state, prices, weights, max_capacity) == -1:
        next_state = state[:]
        while calculate_fitness(next_state, prices, weights, max_capacity) == -1:
            idx = random.randrange(len(next_state))
            if next_state[idx] > 0:
                next_state[idx] -= 1

        neighbours.append(next_state)

    return neighbours


def hill_climbing(initial_state, problem, max_capacity):
    prices = problem["prices"]
    weights = problem["weights"]

    current = initial_state

    best_score = calculate_fitness(current, prices, weights, max_capacity)
    best_state = current

    while True:
        for nbr in next_states(current, problem, max_capacity):
            score = calculate_fitness(nbr, prices, weights, max_capacity)
            if score > best_score:
                best_score = score
                best_state = nbr

        if best_state == current:
            return best_state

        current = best_state


###########################
""" Simulated annealing """
###########################


def simulated_annealing(initial_state, problem, max_capacity, max_iteration=2000):
    prices = problem["prices"]
    weights = problem["weights"]

    current = initial_state
    current_score = calculate_fitness(current, prices, weights, max_capacity)
    temperature = 1000.0
    best_state = current
    best_score = current_score

    for i in range(max_iteration):
        possible_nexts = next_states(current, problem, max_capacity)
        next_state = random.choice(possible_nexts)

        score = calculate_fitness(next_state, prices, weights, max_capacity)

        delta = score - current_score
        tolerance = np.exp(-delta / temperature) 

        if score > current_score or random.uniform(0, 1) < tolerance:
            current = next_state
            current_score = score

        if score > best_score:
            best_state = next_state
            best_score = score

        # Decrease temperature
        temperature *= 0.99

    return best_state

if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Solve knapsnack problem using Genetic Algorithm, Hill Climbing or Simulated Annealing"
    )

    parse.add_argument(
        "--algorithm",
        choices=["ga", "hc", "sa"],
        help="Choose from genetic algorithm (ga), hill climbing (hc) or simulated annealing (sa)",
        action="store",
        required=True,
    )
    parse.add_argument(
        "--file", help="Path to file containing items", action="store", required=True
    )

    args = parse.parse_args(sys.argv[1:])

    try:
        max_capacity, problem = load_knapsack_problem(args.file)

        if args.algorithm == "ga":
            initial_population = generate_population(
                problem["counts"], population_size=200
            )

            solution = genetic_algorithm(problem, initial_population, max_capacity)

        elif args.algorithm == "hc":
            init_state = np.random.randint(2, size=len(problem["names"])).tolist()
            solution = hill_climbing(init_state, problem, max_capacity)

        elif args.algorithm == "sa":
            init_state = np.random.randint(2, size=len(problem["names"])).tolist()
            solution = simulated_annealing(init_state, problem, max_capacity)

        fitness = calculate_fitness(
            solution, problem["prices"], problem["weights"], max_capacity
        )

        print()
        for item_name, optimal_amount, item_value in zip(
            problem["names"], solution, problem["prices"]
        ):
            if optimal_amount:
                print(
                    f"{item_name:7s} [ {optimal_amount:2d} * {item_value:.2f} $ ] -> {item_value * optimal_amount:.2f} $"
                )

        print(f"\nTotal price: {fitness:.2f} $")

    except FileNotFoundError:
        print(f"Can't find file '{args.file}'")

    except ValueError as e:
        print("Unexpected file content.")
        print(e)
