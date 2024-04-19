import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from random import randint
import pdb

matplotlib.use('TkAgg')

def euc_dist(graph, i, j):
    x0, y0 = nx.get_node_attributes(graph, 'pos')[i]
    x1, y1 = nx.get_node_attributes(graph, 'pos')[j]

    return np.sqrt((x0-x1)**2+(y0-y1)**2)

h = nx.random_geometric_graph(100, 0.2, seed=896803)

for i, j in h.edges():
    #h[i][j]['weight'] = np.abs(np.round(100*np.random.normal(0,1))/100)
    h[i][j]['weight'] = np.round(100*euc_dist(h, i, j))/100
    

pos = nx.get_node_attributes(h, 'pos')
#pdb.set_trace()
#path = nx.shortest_path(h, source=list(h.nodes())[0], target=list(h.nodes())[-1], weight='weight')
path = nx.shortest_path(h, source=77, target=25, weight='weight')
edges_path = list(zip(path,path[1:]))

plt.figure(figsize=(10, 10))
weight_labels = nx.get_edge_attributes(h,'weight')
edges_path_reversed = [(y,x) for (x,y) in edges_path]
edges_path = edges_path + edges_path_reversed

edge_colors = ['black' if not edge in edges_path else 'red' for edge in h.edges()]
nx.draw(h, pos=pos, 
        node_color='green',
        edge_color=edge_colors,
        node_size=60)
#nx.draw_networkx_edge_labels(h,pos,edge_labels=weight_labels)

plt.show()