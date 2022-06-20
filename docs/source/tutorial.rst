Tutorial
===================

Module usage
--------------

Works for both :ref:`Project installation <installation:Project usage installation>` and :ref:`Package installation <installation:Project usage installation>`.

GOOD datasets
^^^^^^^^^^^^^^^^^

There are two ways to import 8 GOOD datasets with 14 domain selections and totally 42 splits:

.. code-block:: python

   # Directly import
   from GOOD.data.good_datasets.good_hiv import GOODHIV
   hiv_datasets, hiv_meta_info = GOODHIV.load(dataset_root, domain='scaffold', shift='covariate', generate=False)
   # Or use register
   from GOOD import register as good_reg
   hiv_datasets, hiv_meta_info = good_reg.datasets['GOODHIV'].load(dataset_root, domain='scaffold', shift='covariate', generate=False)
   cmnist_datasets, cmnist_meta_info = good_reg.datasets['GOODCMNIST'].load(dataset_root, domain='color', shift='concept', generate=False)


GOOD GNNs
^^^^^^^^^^^^^
The best and fair way to compare algorithms with the leaderboard is to use the same and similar graph encoder structure;
therefore, we provide GOOD GNN apis to support. Here, we use an objectified dictionary `config` to pass parameters. More
details about the config: [Document of config (pending)]()

*To use exact GNN*

.. code-block:: python

   from GOOD.networks.models.GCNs import GCN
   model = GCN(config)
   # Or
   from GOOD import register as good_reg
   model = good_reg.models['GCN'](config)

*To only use parts of GNN*

.. code-block:: python

   from GOOD.networks.models.GINvirtualnode import GINEncoder
   encoder = GINEncoder(config)


GOOD algorithms
^^^^^^^^^^^^^^^^^
Try to apply OOD algorithms to your own models?

.. code-block:: python

   from GOOD.ood_algorithms.algorithms.VREx import VREx
   ood_algorithm = VREx(config)
   # Then you can provide it to your model for necessary ood parameters,
   # and use its hook-like function to process your input, output, and loss.

Project usage
-----------------

Please refer to :ref:`Project installation <installation:Project usage installation>` for installation details.

Quick intro
^^^^^^^^^^^^^^^

It is a good beginning to directly make it work. Here, we provide command line script `goodtg` (GOOD to go) to access the main function located at `GOOD.kernel.pipeline:main`.
Choosing a config file in `configs/GOOD_configs`, we can start a task:

.. code-block:: shell

   goodtg --config_path GOOD_configs/GOODCMNIST/color/concept/DANN.yaml


Specifically, the task is clearly divided into three steps:

1. **Config**

.. code-block:: python

   from GOOD import config_summoner
   from GOOD.utils.args import args_parser
   from GOOD.utils.logger import load_logger
   args = args_parser()
   config = config_summoner(args)
   load_logger(config)

2. **Loader**

.. code-block:: python

   from GOOD.kernel.pipeline import initialize_model_dataset
   from GOOD.ood_algorithms.ood_manager import load_ood_alg
   model, loader = initialize_model_dataset(config)
   ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

*Or concretely,*

.. code-block:: python

   from GOOD.data import load_dataset, create_dataloader
   from GOOD.networks.model_manager import load_model
   from GOOD.ood_algorithms.ood_manager import load_ood_alg
   dataset = load_dataset(config.dataset.dataset_name, config)
   loader = create_dataloader(dataset, config)
   model = load_model(config.model.model_name, config)
   ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

3. **Train/test pipeline**

.. code-block:: python

   from GOOD.kernel.pipeline import load_task
   load_task(config.task, model, loader, ood_algorithm, config)

*Or concretely,*

.. code-block:: python

   # Train
   from GOOD.kernel.train import train
   train(model, loader, ood_algorithm, config)
   # Test
   from GOOD.kernel.evaluation import evaluate
   test_stat = evaluate(model, loader, ood_algorithm, 'test', config)


How to use this project
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Customization**

To make full use of the project, we can add or modify datasets, GNNs, and OOD algorithms in :mod:`GOOD.data.good_datasets`,
:mod:`GOOD.networks.models`, and :mod:`GOOD.ood_algorithms.algorithms`, respectively. You may resort to :doc:`custom` for more details.

**Understand configs**

Except for customization, an important step is to understand how arguments are passed to where you need. The :doc:`configs`
describes the GOOD way for configurations.

**Run the project**

With added config files (*e.g.*, my_configs/my_datasets/XXX/my_algorithm_config1.yaml), one can run the project on
index 2 GPU.

.. code-block:: shell

   goodtg --config_path my_configs/my_datasets/XXX/my_algorithm_config1.yaml --gpu_idx 2

