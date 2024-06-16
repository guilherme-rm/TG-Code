import networkx as nx
import matplotlib.pyplot as plt
import pdb
from grid import *


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20
})

######################################

G = nx.karate_club_graph()

nx.draw(G, with_labels=True, font_family='serif')
save_fig('zachary')


G_dir = nx.DiGraph(
    [
        ("5", "0"),
        ("0", "1"),
        ("0", "4"),
        ("1", "2"),
        ("1", "3"),
        ("3", "4"),
        ("5", "2"),
        ("5", "6"),
        ("7", "5"),
    ]
)

for layer, nodes in enumerate(nx.topological_generations(G_dir)):
    for node in nodes:
        G_dir.nodes[node]["layer"] = layer

pos = nx.multipartite_layout(G_dir, subset_key="layer")

fig, ax = plt.subplots()
nx.draw(G_dir, pos=pos, ax=ax, font_family='serif', with_labels=True)
ax.set_title("Example of directed graph")
save_fig('directed')


######################################

G_un = nx.Graph(
    [
        ("5", "0"),
        ("0", "1"),
        ("0", "4"),
        ("1", "2"),
        ("1", "3"),
        ("3", "4"),
        ("5", "2"),
        ("5", "6"),
        ("7", "5"),
    ]
)

fig, ax = plt.subplots()
nx.draw(G_un, pos=pos, ax=ax, font_family='serif', with_labels=True)

ax.set_title("Example of undirected graph")
save_fig('undirected')

######################################

G = nx.Graph()

G.add_edge("0", "1", weight=0.6)
G.add_edge("0", "2", weight=0.2)
G.add_edge("2", "3", weight=0.1)
G.add_edge("2", "4", weight=0.7)
G.add_edge("2", "5", weight=0.9)
G.add_edge("0", "3", weight=0.3)

pos = nx.spring_layout(G, seed=8) 

fig, ax = plt.subplots()
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw(G, pos=pos, ax=ax, font_family='serif', with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels)
ax.set_title("Example of weighted graph")
save_fig('weighted')

######################################

n = 5
mu = 1
sigma = 0.5

g = generate_grid(n)

pos = list(g.nodes())[0]
pos_open = list(g.nodes())[0]
target = list(g.nodes())[-1]
cost = 0
cost_open = 0
i = 0
frames = 0
path0 = get_shortest_path(g, pos, target)

draw_graph(g, path0, path0)

save_fig('shortest_path')

#############################################

n = 5
mu = 1
sigma = 0.5

g = generate_grid(n)

pos = list(g.nodes())[0]
pos_open = list(g.nodes())[0]
target = list(g.nodes())[-1]
cost = 0
cost_open = 0
i = 0
frames = 0
path0 = get_shortest_path(g, pos, target)

draw_graph(g, path0, path0, False)

save_fig('grid_ex')
