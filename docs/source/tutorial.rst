Tutorial
===================

Quick intro
---------------

Run an algorithm
^^^^^^^^^^^^^^^^^^

It is a good beginning to make it work directly. Here, we provide the CLI `goodtg` (GOOD to go) to
access the main function located at `GOOD.kernel.main:goodtg`.
Choosing a config file in `configs/GOOD_configs`, we can start a task:

.. code-block:: shell

   goodtg --config_path GOOD_configs/GOODCMNIST/color/concept/DANN.yaml

Hyperparameter sweeping
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform automatic hyperparameter sweeping and job launching, you can use `goodtl` (GOOD to launch):

.. code-block:: shell

   goodtl --sweep_root sweep_configs --launcher MultiLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_devices 0 1 2 3


* `--sweep_root` is a config fold located at `configs/sweep_configs`, where we provide a GSAT algorithm hyperparameter sweeping setting example (on GOODMotif dataset, basis domain, and covariate shift).
  * Each hyperparameter searching range is specified by a list of values. [Example](/../../blob/GOODv1/configs/sweep_configs/GSAT/base.yaml)
  * These hyperparameter configs will be transformed to be CLI argument combinations.
  * Note that hyperparameters in inner config files will overwrite the outer ones.
* `--launcher` denotes the chosen job launcher. Available launchers:
  * `Launcher`: Dummy launcher, only print.
  * `SingleLauncher`: Sequential job launcher. Choose the first device in `--allow_devices`.
  * `MultiLauncher`: Multi-gpu job launcher. Launch on all gpus specified by `--allow_devices`.
* `--allow_XXX` denotes the job scale. Note that for each "allow" combination (e.g. GSAT GOODMotif basis covariate),
there should be a corresponding sweeping config: `GSAT/GOODMotif/basis/covaraite/base.yaml` in the fold specified
by `--sweep_root`.
* `--allow_devices` specifies the gpu devices used to launch jobs.

Sweeping result collection and config update.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To harvest all fruits you have grown (collect all results you have run), please use `goodtl` with a special launcher `HarvestLauncher`:

.. code-block:: shell

   goodtl --sweep_root sweep_configs --final_root final_configs --launcher HarvestLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT


* `--sweep_root`: We still need it to specify the experiments that can be harvested.
* `--final_root`: A config store place that will store the best config settings.
We will update the best configurations (according to the sweeping) into the config files in it.

(Experimental function.)

The output numpy array:
* Rows: In-distribution train/In-distribution test/Out-of-distribution train/Out-of-distribution test/Out-of-distribution validation
* Columns: Mean/Std.

Final runs
^^^^^^^^^^^

It is sometimes not practical to run 10 rounds for hyperparameter sweeping, especially when the searching space is huge.
Therefore, we can generally run hyperparameter sweeping for 2~3 rounds, then perform all rounds after selecting the best hyperparameters.
Now, remove the `--sweep_root`, set `--config_root` to your updated best config saving location, and set the `--allow_rounds`.

.. code-block:: shell

   goodtl --config_root final_configs --launcher MultiLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_devices 0 1 2 3 --allow_rounds 1 2 3 4 5 6 7 8 9 10


Note that the results are valid only after 3+ rounds experiments in this benchmark.

Final result collection
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   goodtl --config_root final_configs --launcher HarvestLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_rounds 1 2 3 4 5 6 7 8 9 10


(Experimental function.)

The output numpy array:
* Rows: In-distribution train/In-distribution test/Out-of-distribution train/Out-of-distribution test/Out-of-distribution validation
* Columns: Mean/Std.

You can customize your own launcher at `GOOD/kernel/launchers/`.


GOOD modules
--------------

GOOD datasets
^^^^^^^^^^^^^^^^^

There are two ways to import 11 GOOD datasets with 17 domain selections:

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
therefore, we provide GOOD GNN APIs to support. Here, we use an objectified dictionary `config` to pass parameters. More
details about the config: :doc:`configs`

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



Deep into details (Preparations for adding new algorithms)
--------------------------------------------------------------

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

   from GOOD.kernel.main import initialize_model_dataset
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

   from GOOD.kernel.pipeline_manager import load_pipeline
   pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
   pipeline.load_task()

After that, the loaded `pipeline` instance will take over the training and test process. The default pipeline is `Pipeline`
defined in :mod:`GOOD.kernel.pipelines.basic_pipeline`. Generally, it is not necessary to modify the pipeline to add new algorithms,
but we allow you to create your own pipelines by creating a pipeline class and registering it:

.. code-block:: python

   @register.pipeline_register
   class YourPipeline:
       pass



How to use this project
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Customization**

To make full use of the project, we can add or modify datasets, GNNs, and OOD algorithms in :mod:`GOOD.data.good_datasets`,
:mod:`GOOD.networks.models`, and :mod:`GOOD.ood_algorithms.algorithms`, respectively. You may resort to :doc:`custom` for more details.

**Understand configs**

Except for customization, an important step is to understand how arguments are passed to where they are needed. The :doc:`configs`
describes the GOOD way for configurations.


