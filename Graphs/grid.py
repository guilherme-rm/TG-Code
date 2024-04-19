import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from random import randint
import pdb
import torch

matplotlib.use('TkAgg')

def weight_nodes(graph: nx.Graph, mu=0.0, sigma=1.0, seed=42, init=False):
    np.random.seed(seed)
    if init:
        for node in graph.nodes():
            graph.nodes[node]['weight'] = [(np.abs(np.round(100*np.random.normal(mu, sigma))/100))]
    else:
        for node in graph.nodes():
            graph.nodes[node]['weight'].append(np.abs(np.round(100*np.random.normal(mu, sigma))/100))

    weight_edges(graph)

def weight_edges(graph: nx.Graph):
    for i,j in graph.edges():
        graph[i][j]['weight'] = np.round(100*(graph.nodes[i]['weight'][-1] + graph.nodes[j]['weight'][-1])/2)/100

def generate_grid(n: int, seed=42):
    g = nx.grid_2d_graph(n, n)
    weight_nodes(g, seed=seed, init=True)
    return g

def get_shortest_path(g:nx.Graph, source, target):
    path = nx.shortest_path(g, source=source, target=target, weight='weight')
    edges_path = list(zip(path,path[1:]))
    return edges_path

def draw_graph(g, path0, path1):
    plt.figure(figsize=(10, 10))
    pos = {(x,y):(y,-x) for x,y in g.nodes()}
    weight_labels = nx.get_edge_attributes(g,'weight')
    edge_colors = []
    for edge in g.edges():
        if edge in path0:
            edge_colors.append('red')
        elif edge in path1:
            edge_colors.append('blue')
        else: edge_colors.append('black')

    nx.draw(g, pos=pos, 
            node_color='lightgreen',
            edge_color=edge_colors, 
            with_labels=True,
            node_size=600)
    nx.draw_networkx_edge_labels(g,pos,edge_labels=weight_labels)

    plt.show()

def draw_graph_anim(ax, g, path0, path1):
    
    pos = {(x,y):(y,-x) for x,y in g.nodes()}
    weight_labels = nx.get_edge_attributes(g,'weight')
    edge_colors = []
    for edge in g.edges():
        if edge in path0:
            edge_colors.append('red')
        elif edge in path1:
            edge_colors.append('blue')
        else: edge_colors.append('black')

    nx.draw(g, pos=pos, 
            node_color='lightgreen',
            edge_color=edge_colors, 
            with_labels=True,
            node_size=600,
            ax=ax)
    nx.draw_networkx_edge_labels(g,pos,edge_labels=weight_labels, ax=ax)

    plt.show()

def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()