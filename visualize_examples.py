import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from torch_geometric.data import Batch

from GOOD import config_summoner, args_parser
from GOOD.data import load_dataset
from GOOD.definitions import STORAGE_DIR, ROOT_DIR
from GOOD.utils.graph_visualize import plot_calculation_graph
from GOOD.utils.initial import reset_random_seed


# from PIL import Image


def plot_molecule(graph, DEBUG, figure_save_path):
    mol = Chem.MolFromSmiles(graph.smiles)
    fig, axes = plt.subplots(dpi=300)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.001, hspace=.001)

    mol_img = MolToImage(mol, size=(600, 600))
    mol_img = mol_img.convert('RGBA')
    update_data = []
    for pixel in mol_img.getdata():
        if pixel[:3] == (255, 255, 255):
            update_data.append((255, 255, 255, 0))
        else:
            update_data.append(pixel)
    mol_img.putdata(update_data)

    axes.axis('off')
    axes.imshow(mol_img)
    if DEBUG:
        fig.show()
    else:
        dir_name = os.path.dirname(figure_save_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig.savefig(figure_save_path, transparent=True)
    plt.close(fig)


def plot_graph(dataset_name, graph, figure_save_path, DEBUG, **kwargs):
    label_attr = None
    if 'Motif' in dataset_name:
        color_attr = 'node_gt'
        graph.node_gt[graph.node_gt == 0] = 5
        graph.node_gt[graph.node_gt == 1] = 8
    elif 'SST2' in dataset_name:
        color_attr = None
        label_attr = 'sentence_tokens'
    else:
        color_attr = 'x'
        graph.x[(graph.x < 0.1).sum(1) > 2] = torch.tensor([0.5, 0.5, 0.5])

    if DEBUG:
        figure_save_path = None
    plot_calculation_graph(graph, color_attr=color_attr,
                           enable_label=True if label_attr is not None else False,
                           font_color='black' if label_attr is not None else 'white',
                           font_size=18,
                           enable_colorbar=False,
                           node_size=350,
                           line_width=1.5,
                           arrows=False,
                           pos=graph.get('pos'),
                           label_attr=label_attr,
                           # vmin=graph.get(color_attr).min() if color_attr is not None else None,
                           # vmax=graph.get(color_attr).max() if color_attr is not None else None,
                           vmin=0,
                           vmax=20,
                           save_fig_path=figure_save_path, **kwargs)


def plot_dataset(i, dataset, dataset_figure_path, DEBUG, shift_type, config):
    dataset_name = config.dataset.dataset_name
    if config.model.model_level == 'graph':
        if shift_type == 'covariate':
            for set_name in ['train', 'test']:
                for graph_id in range(100, 130):
                    figure_save_path = os.path.join(dataset_figure_path, shift_type, set_name, f'{graph_id}.png')
                    graph = dataset[set_name][graph_id]

                    if config.dataset.dataset_type == 'mol':
                        plot_molecule(graph, DEBUG, figure_save_path)
                    else:
                        plot_graph(dataset_name, graph, figure_save_path, DEBUG)
        else:
            mean_y = np.nanmean(torch.cat([dataset['train'].data.y, dataset['test'].data.y]))
            if 'Motif' in dataset_name:
                if 'basis' in dataset_name:
                    domain = 'basis_id'
                else:
                    domain = 'width_id'
                dataset['train'].data.domain_id = dataset['train'].data.get(domain)
                dataset['test'].data.domain_id = dataset['test'].data.get(domain)
                domain_choice = dataset['train'].data.domain_id.unique()
            elif 'CMNIST' in dataset_name:
                dataset['train'].data.domain_id = dataset['train'].data.color
                dataset['test'].data.domain_id = dataset['test'].data.color
                domain_choice = dataset['train'].data.domain_id.unique()
            else:
                domain_intersection = np.intersect1d(dataset['train'].data.domain_id.unique(),
                                                     dataset['test'].data.domain_id.unique())
                domain_choice = np.random.choice(domain_intersection, 10)
            for domain in domain_choice:
                train_samples = torch.nonzero(dataset['train'].data.domain_id == domain).reshape(-1)
                test_samples = torch.nonzero(dataset['test'].data.domain_id == domain).reshape(-1)
                found = False
                for train_id in train_samples:
                    for test_id in test_samples:
                        train_graph = dataset['train'][train_id]
                        test_graph = dataset['test'][test_id]
                        if config.dataset.dataset_type == 'syn':
                            if train_graph.y != test_graph.y:
                                found = True
                        else:
                            if (np.nanmean(train_graph.y) - mean_y) * (np.nanmean(test_graph.y) - mean_y) < 0:
                                found = True
                        if found:
                            name_too_long = False
                            if len(train_graph.y.shape) > 1 and train_graph.y.shape[1] > 10:
                                name_too_long = True
                            if name_too_long:
                                figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'train',
                                                                f'figure.png')
                                figure_save_dir = os.path.dirname(figure_save_path)
                                if not os.path.exists(figure_save_dir):
                                    os.makedirs(figure_save_dir)
                                with open(os.path.join(figure_save_dir, f'label.txt'), 'w') as f:
                                    f.write(f'{train_graph.y}')
                            else:
                                figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'train',
                                                                f'{train_graph.y}.png')

                            if config.dataset.dataset_type == 'mol':
                                plot_molecule(train_graph, DEBUG, figure_save_path)
                            else:
                                plot_graph(dataset_name, train_graph, figure_save_path, DEBUG)

                            if name_too_long:
                                figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'test',
                                                                f'figure.png')
                                figure_save_dir = os.path.dirname(figure_save_path)
                                if not os.path.exists(figure_save_dir):
                                    os.makedirs(figure_save_dir)
                                with open(os.path.join(figure_save_dir, f'label.txt'), 'w') as f:
                                    f.write(f'{test_graph.y}')
                            else:
                                figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'test',
                                                                f'{test_graph.y}.png')
                            if config.dataset.dataset_type == 'mol':
                                plot_molecule(test_graph, DEBUG, figure_save_path)
                            else:
                                plot_graph(dataset_name, test_graph, figure_save_path, DEBUG)
                        if found:
                            break
                    if found:
                        break

    else:
        graph = dataset[0]
        if 'Cora' in dataset_name or 'Arxiv' in dataset_name:
            color_attr = graph.get('domain')
        else:
            color_attr = 'domain_id'
        if shift_type == 'covariate':
            for set_name in ['train', 'test']:
                TRAIN = True if set_name == 'train' else False
                if TRAIN:
                    set_idx = torch.nonzero(graph.train_mask)
                    sort_attr = torch.sort(graph.get(color_attr)[set_idx].squeeze())
                    pick_idx = set_idx[sort_attr.indices[-10:]]
                    if 'Arxiv_degree' in dataset_name:
                        pick_idx = set_idx[sort_attr.indices[-100:-90]]
                else:
                    set_idx = torch.nonzero(graph.test_mask)
                    sort_attr = torch.sort(graph.get(color_attr)[set_idx].squeeze())
                    pick_idx = set_idx[sort_attr.indices[:10]]
                for node_idx in pick_idx:
                    figure_save_path = os.path.join(dataset_figure_path, shift_type, set_name, f'{node_idx}.png')
                    if DEBUG:
                        figure_save_path = None
                    plot_calculation_graph(graph, color_attr=color_attr,
                                           enable_label=False,
                                           enable_colorbar=False,
                                           node_size=50,
                                           line_width=0.5,
                                           arrows=False,
                                           k_hop=2,
                                           graph_idx=node_idx,
                                           vmin=graph.get(color_attr).min(),
                                           vmax=graph.get(color_attr).max(),
                                           save_fig_path=figure_save_path)
        else:
            mean_y = np.nanmean(graph.y)
            if config.dataset.dataset_type == 'syn':
                domain_choice = graph.domain_id.unique()
            else:
                domain_intersection = np.intersect1d(graph.domain_id[graph.train_mask].unique(),
                                                     graph.domain_id[graph.test_mask].unique())
                domain_choice = np.random.choice(domain_intersection, 10)
            for domain in domain_choice:
                train_samples = torch.nonzero((graph.domain_id == domain) & graph.train_mask).reshape(-1, 1)
                test_samples = torch.nonzero((graph.domain_id == domain) & graph.test_mask).reshape(-1, 1)
                found = False
                for train_id in train_samples:
                    for test_id in test_samples:
                        if config.dataset.dataset_type == 'syn':
                            if graph.y[train_id] != graph.y[test_id]:
                                found = True
                        else:
                            if (np.nanmean(graph.y[train_id]) - mean_y) * (np.nanmean(graph.y[test_id]) - mean_y) < 0:
                                found = True
                        if found:
                            figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'train',
                                                            f'{graph.y[train_id]}.png')
                            if DEBUG:
                                figure_save_path = None
                            plot_calculation_graph(graph, color_attr=color_attr,
                                                   enable_label=False,
                                                   enable_colorbar=False,
                                                   node_size=50,
                                                   line_width=0.5,
                                                   arrows=False,
                                                   k_hop=2,
                                                   graph_idx=train_id,
                                                   vmin=graph.get(color_attr).min(),
                                                   vmax=graph.get(color_attr).max(),
                                                   save_fig_path=figure_save_path)

                            figure_save_path = os.path.join(dataset_figure_path, shift_type, f'{domain}', 'test',
                                                            f'{graph.y[test_id]}.png')
                            if DEBUG:
                                figure_save_path = None
                            plot_calculation_graph(graph, color_attr=color_attr,
                                                   enable_label=False,
                                                   enable_colorbar=False,
                                                   node_size=50,
                                                   line_width=0.5,
                                                   arrows=False,
                                                   k_hop=2,
                                                   graph_idx=test_id,
                                                   vmin=graph.get(color_attr).min(),
                                                   vmax=graph.get(color_attr).max(),
                                                   save_fig_path=figure_save_path)
                        if found:
                            break
                    if found:
                        break

@torch.no_grad()
def plot_interpretable_graphs(i, dataset, dataset_figure_path, DEBUG, shift_type, config):
    from GOOD.kernel.main import initialize_model_dataset, load_ood_alg, load_pipeline
    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
    pipeline.config_model('test', load_param=True)
    model = pipeline.model
    model.eval()
    dataset_name = config.dataset.dataset_name
    if config.model.model_level == 'graph':
        if shift_type == 'covariate':
            for set_name in ['test']:#['train', 'test']:
                for graph_id in range(100, 150):
                    graph = dataset[set_name][graph_id]
                    graph = Batch.from_data_list([graph]).to(config.device)
                    graph.node_gt[torch.where(graph.node_gt == 1)[0][:2]] = 0

                    model_output = model(data=graph, ood_algorithm=pipeline.ood_algorithm)
                    interpretable_mask = model.edge_mask.squeeze()
                    for mask_ratio in [0.3, 0.4, 0.5, 0.7, 0.9, 0.95]:
                        figure_save_path = os.path.join(dataset_figure_path, shift_type, set_name, f'{graph_id}_{mask_ratio}.png')
                        edge_color_matrix = torch.zeros((graph.x.shape[0], graph.x.shape[0]), dtype=torch.bool,
                                                        device=config.device)
                        if interpretable_mask.dim() == 2:
                            edge_color_matrix[interpretable_mask[0], interpretable_mask[1]] = True
                        else:
                            edge_color = interpretable_mask > 0 if (interpretable_mask > 1).sum() > 0 else interpretable_mask > mask_ratio
                            # if set_name == 'test':
                            #     edge_color.fill_(False)
                            #     edge_color[interpretable_mask.topk(6).indices] = True
                            edge_color_matrix[graph.edge_index[0], graph.edge_index[1]] = edge_color
                        # To make symmetric edges has the same color
                        edge_color_matrix = edge_color_matrix | edge_color_matrix.T
                        edge_color = edge_color_matrix[graph.edge_index[0], graph.edge_index[1]].float()
                        edge_color = edge_color * 0.35   # color adjustment


                        if config.dataset.dataset_type == 'mol':
                            plot_molecule(graph, DEBUG, figure_save_path)
                        else:
                            plot_graph(dataset_name, graph, figure_save_path, DEBUG, edge_color=edge_color)


DEBUG = False

allowed_datasets = ['GOODMotif']
allowed_methods = ['GEI'] #['ASAP', 'DIR', 'GSAT', 'CIGA', 'GEI']
config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'final_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    if dataset_path.name not in allowed_datasets:
        continue
    # single_dataset_paths = []
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir() or domain_path.name == 'size':
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            if shift_path.name != 'covariate':
                continue
            for ood_config_path in shift_path.iterdir():
                if ood_config_path.stem in allowed_methods:
                    # single_dataset_paths.append(str(ood_config_path))
                    config_paths.append(str(ood_config_path))



plt.grid(False)

from tqdm import tqdm
from GOOD.utils.logger import pbar_setting

pbar = tqdm(enumerate(config_paths), total=len(config_paths), **pbar_setting)
for i, config_path in pbar:
    args = args_parser(['--config_path', config_path])
    config = config_summoner(args)
    reset_random_seed(config)
    pbar.set_description(f"{config.dataset.dataset_name} {config.dataset.shift_type}")
    dataset = load_dataset(config.dataset.dataset_name, config)
    # continue
    dataset_figure_path = os.path.join(STORAGE_DIR, 'figures', 'dataset_examples', f'{config.dataset.dataset_name}', f'{config.ood.ood_alg}')
    plot_interpretable_graphs(i, dataset, dataset_figure_path, DEBUG, config.dataset.shift_type, config)
    # plot_dataset(i, dataset, dataset_figure_path, DEBUG, config.dataset.shift_type, config)
    # exit(0)
