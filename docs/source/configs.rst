Configs and command line arguments
============================================

There are always many configuration dilemmas in managing code running configurations.

Traditional command line arguments
------------------------------------

When we only use command line arguments as running configs, it is always annoying to run code like this:

.. code-block:: shell

   python run_my_code.py --model GCN --layers 5 --dim_hidden 300 --save_dir xx/xx ......

These running commands bring difficulties running batches of experiments and achieving easily reproducibility.

Traditional yaml-file-based configs
-------------------------------------------

It is a better idea for data scientists to adopt reading dictionaries from yaml files as their parameters storage strategy. However,
when the configuration scale becomes large, we will face a hard time remembering all the names of configs, so
that we can access the config like this:

.. code-block:: python

   config: dict = config_loader(file_path)
   model_layer: int = config['model']['model_layer']
   model_name: str = config['model']['model_name']
   ...

It is fine if we can remember all details of these parameters, such as their names, their types, and the attributes of them.

If we can't, here is the strategy we design.

GOOD Configs and command line Arguments (CA)
-----------------------------------------------

There are several advantages of using our configuration strategy.

- Convenient: GOOD CA allows reading configs from yaml files with :ref:`include <configs:include>` support.
- Flexible: GOOD CA enables overwriting specific reading configs by passing command line arguments.
- Diversified access & code-complete support: With the help of `Munch <https://github.com/Infinidat/munch>`_, we can
access configs in both dictionary and objective ways. In GOOD, configs and command line arguments are coherent connected;
therefore, by defining command line arguments, the names, types, and attributes of configs can be easily found. Hence,
many plugins and IDEs (*e.g.*, PyCharm) can make use of this connection and provide code complete supports.

Config file
^^^^^^^^^^^^^^

Yaml format is supported to define arguments in config files (*e.g.*, `Coral config <https://github.com/divelab/GOOD/blob/docs/configs/GOOD_configs/GOODCMNIST/color/covariate/Coral.yaml>`_).
There is a special keyword :obj:`include` that LIST all other necessary config file paths. This keyword works like
:obj:`import` command in python.

.. note::
   Local file configs will overwrite imported configs when face key name duplications.

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

Given the config structure shown above, there are two ways to access:

.. code-block:: python

   # dict
   model_name = config['model']['model_name']
   # object
   model_name = config.model.model_name

Command line arguments
^^^^^^^^^^^^^^^^^^^^^^^^^

Command line arguments play a totally different role compared to config files. It provides config file choosing,
arguments overwriting, and code hints for code-complete. In GOOD, we adopt `typed-argument-parser <https://github.com/swansonk14/typed-argument-parser#loading-from-configuration-files>`_
to organize and parse command line arguments.

Arguments passed as command line arguments will overwrite arguments in config files. For example:

.. code-block:: shell

   goodtg --config_path XXX/XXX.yaml --gpu_idx 1

This command will overwrite the config's :obj:`gpu_idx` argument to 1, which implying using the index 1 GPU.

**Command line argument structure**

As config code hints, command line argument structure has a one-to-one corresponding relationship with config structure.

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

.. note::
   There shouldn't be any arguments with the same name, even in different argument classes.

**Code completion & new arguments**

When we connect our configs with the command line arguments, many IDEs will provide code-completion for our configs.

.. code-block:: python

   config: Union[Munch, CommonArgs]
   config.  # It will prompt: random_seed, task, train, model, dataset, etc.
   config.model.  # It will prompt: model_name, model_layer, dim_hidden, etc.

.. warning::
   When you add a **new custom argument** into your config file, you will be warned to add corresponding arguments into
   command line argument class. For example, when you add a argument as :obj:`config.dataset.author`, you should add
   argument :obj:`author` to class :class:`GOOD.utils.args.DatasetArgs`.

How to pass configs to an object (Module usage)
---------------------------------------------------

When we use GOOD for modules, it is still simple to pass configs. Take :class:`GroupDRO <GOOD.ood_algorithms.algorithms.GroupDRO.GroupDRO>`
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