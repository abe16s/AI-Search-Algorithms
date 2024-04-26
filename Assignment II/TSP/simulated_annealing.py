import utils
import random
import math

def generateSuccessors(tour):
	N = len(tour)        
	
	randomSequence1 = list(range(N))
	random.shuffle(randomSequence1)
	randomSequence2 = list(range(N))
	random.shuffle(randomSequence2)
	

	for i in randomSequence1:
		for j in randomSequence2:
			
			#to avoid swapping same pair twice
			if i<j:
				temp = list(tour)
				temp[i],temp[j] = tour[j],tour[i]
				yield temp


def decreasing_probability(current_time, initial_prob, decay_rate):
    prob = initial_prob * math.exp(-decay_rate * current_time)
    return prob


def simulated_annealing(cities, graph, generation):
    path = cities
    random.shuffle(path)

    bestTour =  worstTour = path
    bestValue = worstValue = utils.calculate_cost(path, graph)

    for i in range(generation):

        moved = False
	
        for successor in generateSuccessors(bestTour):
            successorValue = utils.calculate_cost(successor, graph=graph)
			
            if successorValue < bestValue:
                bestTour  = successor
                bestValue = successorValue
                moved = True
                break
            else:
                  worstTour = successor
                  worstValue = successorValue
                        

        if random.uniform(0, 1) < decreasing_probability(i, 1, 0.1):
              bestTour = worstTour
              bestValue = worstValue

        if moved == False:
            break
    return (bestTour, bestValue)

    