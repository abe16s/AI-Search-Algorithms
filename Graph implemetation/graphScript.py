import numpy as np
from io import StringIO
from graph import *
import math

# node_data = np.loadtxt('cities.txt', dtype=str)
# edges_data = np.loadtxt('edges.txt', dtype=str)



# nodes = {}
# for i in node_data:
#     nodes[i[0]] = (i[1], i[2])

# print(nodes)

# edges = []


# for edge in edges_data:
#     lat1, lon1 = nodes[edge[0]]
#     lat2, lon2 = nodes[edge[1]]
#     weight = haversine_distance(lat1=float(lat1), lat2=float(lat2), lon1=float(lon1), lon2=float(lon2))
#     edges.append(Edge(edge[0], edge[1], weight))


# print(edges)
print(haversine_distance(46.166667, 21.316667, 45.792784, 24.1520689999999983))


