import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from random import randint
import pdb
from grid import *
import time as t

matplotlib.use('TkAgg')

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16
})

n = [x for x in range(5, 51)]
mu = 10
sigma = 5

closed_cost = []
closed_time = []
open_cost = []
open_time = []


for num in n:
    g = generate_grid(num)

    pos = list(g.nodes())[0]
    pos_open = list(g.nodes())[0]
    target = list(g.nodes())[-1]
    cost = 0
    cost_open = 0
    i = 0
    frames = 0
    begin = t.time()
    path0 = get_shortest_path(g, pos, target)
    end = t.time()
    open_time.append((end-begin)*1e3)
    t0 = 10

    for k in range(t0):
        weight_nodes(g, mu=mu, sigma=sigma, seed=k)
    curr_closed_time = 0
    while pos != target:
        begin = t.time()
        path = get_shortest_path(g, pos, target)
        curr_closed_time += t.time() - begin
        cost += g[pos][path[0][1]]['weight']
        pos = path[0][1]
        if (pos_open != target):
            cost_open += g[pos_open][path0[i][1]]['weight']
            pos_open = path0[i][1]
            i += 1
        weight_nodes(g, mu=mu, sigma=sigma, seed=i+2)
        frames += 1

    print(f'n = {num}, closed loop = {cost:.2f}, open loop = {cost_open:.2f}')
    closed_time.append(curr_closed_time*1e3)
    closed_cost.append(cost)
    open_cost.append(cost_open)

plt.plot(n, closed_cost, label='Closed loop')
plt.plot(n, open_cost, label='Open loop')
plt.xlabel('Number of nodes')
plt.ylabel('Cost')
plt.legend()
plt.grid()
save_fig('cost')

plt.cla()

plt.plot(n, open_time, label='Open loop')
plt.plot(n, closed_time, label='Closed loop')
plt.xlabel('Number of nodes')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid()
save_fig('time')