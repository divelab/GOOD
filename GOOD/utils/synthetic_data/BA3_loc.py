"""BA3_loc.py
"""

import numpy as np

figsize = (8, 6)


####################################
#
# Experiment utilities
#
####################################
def perturb(graph_list, p, id=None):
    """ Perturb the list of (sparse) graphs by adding/removing edges.
    Args:
        p: proportion of added edges based on current number of edges.
    Returns:
        A list of graphs that are perturbed from the original graphs.
        :param graph_list:
        :param id:
    """
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_count = int(G.number_of_edges() * p)
        # randomly add the edges between a pair of nodes without an edge.
        for _ in range(edge_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u, v)) and (u != v):
                    break
            if (not id == None) and (id[u] == 0 or id[v] == 0):
                G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list
