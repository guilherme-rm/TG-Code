import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from random import randint
import pdb
from grid import *

matplotlib.use('TkAgg')

n = 3
mu = 1
sigma = 0.5

g = generate_grid(5)

pos = list(g.nodes())[0]
pos_open = list(g.nodes())[0]
target = list(g.nodes())[-1]
cost = 0
cost_open = 0
i = 0
frames = 0
path0 = get_shortest_path(g, pos, target)
t0 = 10

for k in range(t0):
    weight_nodes(g, seed=k)


pdb.set_trace()

while pos != target:
    path = get_shortest_path(g, pos, target)
    cost += g[pos][path[0][1]]['weight']
    pos = path[0][1]
    if (pos_open != target):
        cost_open += g[pos_open][path0[i][1]]['weight']
        pos_open = path0[i][1]
        i += 1
    weight_nodes(g, seed=i+1)
    frames += 1
    draw_graph(g, path0, path)

print(f'closed loop = {cost:.2f}, open loop = {cost_open:.2f}')




