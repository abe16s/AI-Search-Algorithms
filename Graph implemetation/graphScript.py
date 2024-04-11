import numpy as np
from io import StringIO
from graph import *
import math

# node_data = np.loadtxt('cities.txt', dtype=str)
# edges_data = np.loadtxt('edges.txt', dtype=str)

def haversine_distance(lat1, lon1, lat2, lon2):
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


