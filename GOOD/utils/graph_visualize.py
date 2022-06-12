import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_networkx, subgraph, k_hop_subgraph


# plt.style.use('seaborn')

def plot_graph(nx_Gs, color_attr=None, colors=None, font_color='white', node_size=300,
               arrows=True, vmin=None, vmax=None, pos=None, line_width=1.0,
               enable_label=True, enable_colorbar=True, save_fig_path=None):
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
        lc = nx.draw_networkx_labels(nx_Gs, pos, font_color=font_color, ax=ax)
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
    if getattr(calculation_graph, 'batch') is not None:
        subset = torch.where(calculation_graph.batch == graph_idx)[0]
    else:
        if k_hop:
            subset = k_hop_subgraph(node_idx=graph_idx, num_hops=k_hop, edge_index=calculation_graph.edge_index)[0]
            if subset.shape[0] > 2000:
                print(f'The graph is too large. {subset.shape[0]}')
                subset = k_hop_subgraph(node_idx=graph_idx, num_hops=1, edge_index=calculation_graph.edge_index)[0]

                # return None
        else:
            subset = torch.arange(calculation_graph.x.shape[0], dtype=torch.long, device=calculation_graph.x.device)
    vis_edge_index = \
        remove_self_loops(subgraph(subset=subset, edge_index=calculation_graph.edge_index, relabel_nodes=True)[0])[0]
    data = Data(edge_index=vis_edge_index)
    if color_attr:
        data.__setattr__(color_attr, calculation_graph.get(color_attr)[subset])

    vis_nx_G = to_networkx(data, node_attrs=[color_attr] if color_attr else None)
    plot_graph(vis_nx_G, color_attr=color_attr, **kwargs)


def plot_explanation(simple_G, CBN):
    nx_Gs = to_networkx(Data(edge_index=simple_G.edge_index, color=simple_G.color, CBN=CBN), node_attrs=['color'],
                        edge_attrs=['CBN'])
    # nx_Gs = nx_Gs.to_undirected()
    # nx_Gs = simple_G

    colors = list(nx.get_node_attributes(nx_Gs, 'color').values())
    N_edge_tuple = [(u, v) for u, v, e in nx_Gs.edges(data=True) if e['CBN'] == 2]
    B_edge_tuple = [(u, v) for u, v, e in nx_Gs.edges(data=True) if e['CBN'] == 1]
    C_edge_tuple = [(u, v) for u, v, e in nx_Gs.edges(data=True) if e['CBN'] == 0]

    pos = nx.kamada_kawai_layout(nx_Gs)
    nx_Gs.nodes.data()
    # nx.draw(nx_Gs, pos, with_labels=True, node_color=colors, cmap=plt.cm.jet)
    nec = nx.draw_networkx_edges(nx_Gs, pos, edgelist=N_edge_tuple, style='--')
    bec = nx.draw_networkx_edges(nx_Gs, pos, edgelist=B_edge_tuple, edge_color='cyan', width=1.5)
    cec = nx.draw_networkx_edges(nx_Gs, pos, edgelist=C_edge_tuple, edge_color='red', width=1.5)
    nc = nx.draw_networkx_nodes(nx_Gs, pos, nodelist=nx_Gs.nodes, node_color=colors, cmap=plt.cm.jet, vmin=0)
    lc = nx.draw_networkx_labels(nx_Gs, pos, font_color="white")
    plt.colorbar(nc)
    plt.show()
