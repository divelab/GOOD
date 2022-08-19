r"""Matplotlib utils for graph plots.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_networkx, subgraph, k_hop_subgraph


# plt.style.use('seaborn')

def plot_graph(nx_Gs, color_attr=None, colors=None, font_color='white', font_size=12, node_size=300,
               arrows=True, vmin=None, vmax=None, pos=None, line_width=1.0,
               enable_label=True, label_attr=None, enable_colorbar=True, save_fig_path=None):
    r"""
    Plot a graph with Matplotlib

    Args:
        nx_Gs (Graph): input graph (:obj:`networkx.Graph`)
        color_attr (str or None): color attribute of nodes
        colors (str or list or None): Node color
        font_color (str): font color in plot
        node_size (int): Size of nodes.
        arrows (bool or None): arrowheads style for drawing. If None, directed graphs draw arrowheads with FancyArrowPatch, while undirected graphs draw edges via LineCollection for speed. If True, draw arrowheads with FancyArrowPatches (bendable and stylish). If False, draw edges using LineCollection (linear and fast).
        vmin (float or None): minimum for node colormap scaling
        vmax (float or None): maximum for node colormap scaling
        pos (list or None): positions values of nodes
        line_width (float): line width in plot
        enable_label (bool): whether to add labels in plot
        enable_colorbar (bool): whether to use colorbar in plot
        save_fig_path (str or None): path to save plot file

    Returns:
        None
    """
    if colors is not None:
        colors = colors
    elif color_attr is not None:
        colors = list(nx.get_node_attributes(nx_Gs, color_attr).values())
    else:
        colors = 'red'


    # plt.clf()
    fig, ax = plt.subplots(dpi=300)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.001, hspace=.001)

    ax.axis('off')
    # fig = plt.figure(1)
    if pos is None:
        pos = nx.kamada_kawai_layout(nx_Gs)
    else:
        pos = pos.numpy()
        pos[:, 1] = - pos[:, 1]
        pos = {i: pos[i] for i in range(pos.shape[0])}
    # nx_Gs.nodes.data()
    # nx.draw(nx_Gs, pos, with_labels=True, node_color=colors, cmap=plt.cm.jet)
    ec = nx.draw_networkx_edges(nx_Gs, pos, edgelist=nx_Gs.edges, node_size=node_size, arrowstyle="<|-", arrows=arrows,
                                ax=ax, width=line_width)
    nc = nx.draw_networkx_nodes(nx_Gs, pos, nodelist=nx_Gs.nodes, node_color=colors, cmap=plt.cm.get_cmap('brg'),
                                vmin=vmin, vmax=vmax,
                                ax=ax, node_size=node_size)
    if enable_label:
        lc = nx.draw_networkx_labels(nx_Gs, pos, labels=nx.get_node_attributes(nx_Gs, label_attr), font_color=font_color, font_size=font_size, ax=ax)
    if enable_colorbar:
        plt.colorbar(nc)
    if save_fig_path:
        dir_name = os.path.dirname(save_fig_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(save_fig_path, transparent=True)
    else:
        fig.show()
    plt.close(fig)


def plot_calculation_graph(calculation_graph, graph_idx=0, color_attr=None, k_hop=None, **kwargs):
    r"""
    Processing of graphs before plot

    Args:
        calculation_graph (Batch): graph data to be processed
        graph_idx (int, list, tuple or torch.Tensor): the central node(s)
        color_attr (str or None): color attribute of nodes
        k_hop (int): number of hops in graph to plot
        **kwargs: key word arguments for the use of :func:`~plot_graph`

    Returns:
        None

    """
    if getattr(calculation_graph, 'batch') is not None:
        subset = torch.where(calculation_graph.batch == graph_idx)[0]
    else:
        if k_hop:
            subset = k_hop_subgraph(node_idx=graph_idx, num_hops=k_hop, edge_index=calculation_graph.edge_index)[0]
            if subset.shape[0] > 1000:
                print(f'The graph is too large. {subset.shape[0]}')
                subset = k_hop_subgraph(node_idx=graph_idx, num_hops=1, edge_index=calculation_graph.edge_index)[0]
                if subset.shape[0] > 1000:
                    return

                # return None
        else:
            subset = torch.arange(calculation_graph.x.shape[0], dtype=torch.long, device=calculation_graph.x.device)
    vis_edge_index = \
        remove_self_loops(subgraph(subset=subset, edge_index=calculation_graph.edge_index, relabel_nodes=True, num_nodes=calculation_graph.x.shape[0])[0])[0]
    data = Data(edge_index=vis_edge_index)
    node_attrs = []
    if color_attr:
        data.__setattr__(color_attr, calculation_graph.get(color_attr)[subset])
        node_attrs.append(color_attr)
    label_attr = kwargs.get('label_attr')
    if label_attr:
        labels = calculation_graph.get(label_attr)
        if type(labels) is list:
            label_subset = [labels[i] for i in subset]
        else:
            label_subset = labels[subset]
        data.__setattr__(label_attr, label_subset)
        node_attrs.append(label_attr)

    vis_nx_G = to_networkx(data, node_attrs=node_attrs)
    plot_graph(vis_nx_G, color_attr=color_attr, **kwargs)
    print('One graph plotted')


