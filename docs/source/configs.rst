Configs and CLI
============================================

There are always many configuration dilemmas in managing code running configurations.

CLI arguments
------------------------------------

When we only use CLI to load running configs:

.. code-block:: shell

   python run_my_code.py --model GCN --layers 5 --dim_hidden 300 --save_dir xx/xx ......

YAML-file-based configs
-------------------------------------------

Reading configurations from YAML files is also a good idea. However, when the configuration scale becomes large,
we will face a hard time remembering all the names of configs:

.. code-block:: python

   config: dict = config_loader(file_path)
   model_layer: int = config['model']['model_layer']
   model_name: str = config['model']['model_name']
   ...

GOOD Config and Argument Setting (CA)
-----------------------------------------------

There are several advantages of using our configuration strategy.

- Convenient: GOOD CA allows reading configs from YAML files with :obj:`include` support.
- Flexible: GOOD CA enables overwriting specific reading configs by passing CLI arguments.
- Diversified access & code-complete support: With the help of `Munch <https://github.com/Infinidat/munch>`_, we can
  access configs in both dictionary and objective ways. In GOOD, configs and CLI arguments are coherently connected;
  therefore, by defining command line arguments, the names, types, and attributes of configs can be easily found. Hence,
  many plugins and IDEs (*e.g.*, PyCharm) can use this connection and provide code-completion support.

Config file
^^^^^^^^^^^^^^

The YAML format is supported to define arguments in config files (*e.g.*, `Coral config <https://github.com/divelab/GOOD/blob/docs/configs/GOOD_configs/GOODCMNIST/color/covariate/Coral.yaml>`_).
There is a special keyword :obj:`include` that LIST all other necessary config file paths. This keyword works like
:obj:`import` command in python.

.. note::
   Local file configs will overwrite imported configs when facing key name duplications.

**Config structure**

.. code-block:: yaml

   # general configs
   random_seed: 123
   task: train
   ...
   # model configs
   models:
     model_name: vGIN
     model_layer: 5
     ...
   # train configs
   train:
     ...
   # dataset configs
   dataset:
     ...
   # OOD configs
   ood:
     ...
   # special configs: generated automatically or generated depending on other configs
   # metric: Metric()  # depends on the chosen dataset
   # train_helper: TrainHelper()  # depends on lr, milestones, etc.

**Access**

Given the config structure shown above, there are two ways to access it:

.. code-block:: python

   # dict
   model_name = config['model']['model_name']
   # object
   model_name = config.model.model_name

CLI arguments
^^^^^^^^^^^^^^^^^^^^^^^^^

CLI arguments play a totally different role compared to config files. It provides config file choosing,
arguments overwriting, and code hints for code-complete. In GOOD, we adopt `typed-argument-parser <https://github.com/swansonk14/typed-argument-parser#loading-from-configuration-files>`_
to organize and parse CLI arguments.

Arguments passed as CLI arguments will overwrite arguments in config files. For example:

.. code-block:: shell

   goodtg --config_path XXX/XXX.yaml --gpu_idx 1

This command will overwrite the config's :obj:`gpu_idx` argument by 1, which implying using the index 1 GPU.

**Command line argument structure**

As config code hints, the CLI argument structure has a corresponding one-to-one relationship with the config structure.
As shown in `GOOD/utils/args.py`:

.. code-block:: python

   # General configs
   class CommonArgs(Tap):
       random_seed: str = None  #: Fixed random seed for reproducibility.
       task: Literal['train', 'test'] = None  #: Running mode.
       ...

       # Connect to model, train. dataset, ood configs.
       train: TrainArgs = None  #: For code auto-complete
       model: ModelArgs = None  #: For code auto-complete
       dataset: DatasetArgs = None  #: For code auto-complete
       ood: OODArgs = None  #: For code auto-complete

       def process_args(self):
           ...  # Parse train, model, dataset, ood arguments.

   # Model configs
   class ModelArgs(Tap):
       model_name: str = None  #: Specify the model name.
       model_layer: int = None  #: Number of GNN layer.
       ...

   # Train configs
   class TrainArgs(Tap):
       ...

   # Dataset configs
   class DatasetArgs(Tap):
       ...

   # OOD configs
   class OODArgs(Tap):
       ...

.. warning::
   There should not be any arguments with the same name, even in different argument classes.

**Code completion & new arguments**

When we connect our configs with the command line arguments, many IDEs will provide code completion for our configs.

.. code-block:: python

   config: Union[Munch, CommonArgs]
   config.  # It will prompt: random_seed, task, train, model, dataset, etc.
   config.model.  # It will prompt: model_name, model_layer, dim_hidden, etc.

.. note::
   When adding a **new custom argument** into a config file, we will be warned to add corresponding arguments into
   the command line argument class. For example, when we add an argument as :obj:`config.dataset.author` in yaml files, we should also add
   argument :obj:`author` to class :class:`GOOD.utils.args.DatasetArgs` in `GOOD/utils/args.py`.

How to obtain the configs instance
-----------------------------------

.. code-block:: python

   # Import the CLI argument parser and the config file reader/processor.
   from GOOD.utils.args import args_parser
   from GOOD import config_summoner

   # Parse the CLI arguments including `--config_path /path/to/config/file.yaml`
   args = args_parser()

   # If there is no command line interface, we can pass `--config_path /path/to/config/file.yaml` directly to args_parser.
   args = args_parser(['--config_path', '/path/to/config/file.yaml'])

   # Use parsed args to load config which should be accessed anywhere in this project.
   config = config_summoner(args)


How to pass configs to an object without using the config from a yaml file.
-----------------------------------------------------------------------------

When we use GOOD modules, we need to build the config dictionary to pass configs. Take :class:`GroupDRO <GOOD.ood_algorithms.algorithms.GroupDRO.GroupDRO>`
as an example. When we use the ``loss_postprocess`` function, there should be ``device``, ``dataset.num_envs``, and ``ood.ood_param``
passed in using ``config`` as mentioned in the docs. Therefore, we can use the function as:

.. code-block:: python

   # Define a config dictionary
   config = {
       device: torch.device('cuda:0'),
       dataset: {
           num_envs: 10
           }
       ood: {
           ood_param: 0.1
           }
       }
   from munch import munchify
   # Pass the munchified config.
   groupdro.loss_postprocess(loss, data, mask, munchify(config))
