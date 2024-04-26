import random
import utils
from collections import deque



def generate_population(cities, size, start_city="Arad"):
    other_cities = [c for c in cities if c != "Arad"]

    population = []

    for _ in range(size):
        chromosome = other_cities[:] 
        random.shuffle(chromosome)
        population.append([start_city] + chromosome + [start_city])

    return population



def sort_population(population, graph):
    population.sort(key=lambda i: utils.calculate_cost(i, graph=graph))

    return population


def pick_parents(population):
    index1 = random.randint(0, len(population) - 1)
    index2 = random.randint(0, len(population) - 1)

    return population[index1], population[index2]



def cross_over(parent1, parent2):
    index1 = random.randint(1, len(parent1) - 2)
    index2 = random.randint(1, len(parent2) - 2)

    child = parent1[:index1] + parent2[index2:]
    
    return child


def mutate(chromosome):
    index1 = random.randrange(1, len(chromosome) - 1)
    index2 = random.randrange(1, len(chromosome) - 1)

    chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]

    return chromosome




def genetic_algorithm(cities, population_size, percent, generation, graph):
    
    
    population = generate_population(cities, population_size)
    
    
    for _ in range(generation):
    
    
        population = sort_population(population, graph=graph)
        amount = int(len(population) * percent)
        left = len(population) - amount


        population = population[:amount]

        children = []
        for _ in range(left):
            parent1, parent2 = pick_parents(population)

            child = cross_over(parent1=parent1, parent2=parent2)

            probability = random.uniform(0, 1)
            if probability < 0.1:
                mutate(child)


            probability = random.uniform(0, 1)
            if probability < 0.05:
                mutate(population)
                

            children.append(child)

        population += children

    sort_population(population, graph=graph)


    path = utils.build_full_path(population[0], graph)    

    cost = utils.calculate_cost(population[0], graph)
    
    return path, cost









