Customization & Add a new OOD algorithm
==========================================


Brief introduction
-----------------------

Custom datasets
^^^^^^^^^^^^^^^^

To customize a new dataset that can be directly embedded into the GOOD pipeline, one must implement a static method
load with necessary meta information and register the dataset. For the ``load`` function, nothing is easier than directly
refer :func:`the code as an example <GOOD.data.good_datasets.good_hiv.GOODHIV.load>`.

*Register your dataset:*

.. warning::
   For a successful register, please always remember to locate your class at a place that will be automatically imported
   once the program starts.

.. code-block:: python

   from GOOD import register

   @register.dataset_register
   class CustomDataset(InMemoryDataset):
       pass

Custom GNNs
^^^^^^^^^^^^^^

For customizing your GNN model, it is a good idea to inherit :class:`GNNBasic <GOOD.networks.models.BaseGNN.GNNBasic>` for flexible
argument reading.

*Register your GNN:*

.. code-block:: python

   from GOOD import register

   @register.model_register
   class MyGNN(GNNBasic):
       pass

Custom OOD algorithms
^^^^^^^^^^^^^^^^^^^^^^

Current OOD algorithms can be implemented by overwriting :class:`input_preprocess <GOOD.ood_algorithms.algorithms.BaseOOD.BaseOODAlg>`. With the help of passing
the algorithm object into your GNN, these processes covered the whole batched training pipeline: :func:`train_batch <GOOD.kernel.pipelines.basic_pipeline.train_batch>`.

.. code-block:: python

   data = data.to(self.config.device)

   self.ood_algorithm.optimizer.zero_grad()

   mask, targets = nan2zero_get_mask(data, 'train', self.config)
   node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
    # data augmentation algorithm may implement input_preprocess.
   data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                        self.model.training,
                                                                        self.config)
   edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None

   # algorithms for changing model structure may implement custom GNNs and output_postprocess. Refer to DANN.
   model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
   # generally for multiple model outputs.
   raw_pred = self.ood_algorithm.output_postprocess(model_output)

   # calculate loss with reduction.
   loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
   # aggregate loss.
   loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config)

   self.ood_algorithm.backward(loss)

*register your OOD algorithm:*

.. code-block:: python

   from GOOD import register

   @register.ood_alg_register
   class MyOODAlgorithm(BaseOODAlg):
       pass


Practical steps to add a new ood algorithm
------------------------------------------

Generally, we can access :obj:`config.ood.ood_param` (a float value) and :obj:`config.ood.extra_param` (a list of hyperparameters: float, bool, str...) to
build our algorithms.

1. Build your model:
    * In the `GOOD/networks/models/` folder, copy a model file (*e.g.*, `DANNs.py`) as `my_algorithm_model.py`.
    * Modifiy the class name.
    * Define your model's modules and the forward function. This forward function should handle both training and evaluation cases.
    * A method with multiple concatenated models should combine them into a top model. Multi-stage and separate optimizations can be handel by your algorithm class which will be introduced in the next step.
    * `GINFeatExtractor` and `vGINFeatExtractor` are the two standard GIN and GIN-virtualnode encoders. We can copy & modify them or access their inner objects, but remember to make sure of a fair comparison.
2. Build your algorithm:
    * In the `GOOD/ood_algorithms/algorithms/` folder, copy an algorithm file (*e.g.*, `DANN.py`) as `my_algorithm.py`.
    * This file is used to control the ood algorithm's training stages, output cleaning (for test prediction), multiple ood loss calculations, optimizations.
    * Function `stage_control` is used to change the training stage, *e.g.*, we may pre-train part of the model at the first stage and train the whole model at the second stage.
    * Function `output_postprocess` is used to output only the model logits or regression value for evaluations. In this function, other output should be saved by your algorithm for loss calculations.
    * Function `loss_calculate` and `loss_postprocess` are both designed for loss calculation. The first one is used to calculate only the prediction loss without any special OOD constrains. The second one is used to calculate special OOD constrains. This two functions may be merged into one in the future.
    * Function `set_up` and `backward` serve for optimization designs.
3. Build your config files:
    * Before running, the new algorithm needs its config files. If we want to run GOOD-SST2 dataset's length-covariate split, in `configs/GOOD_configs/length/covariate/` folder, copy a config file (*e.g.*, `GSAT.yaml`) as `my_algorithm.yaml`.
    * Change your `model_name`, `ood_param`, `extra_param`, and other configs.
4. Run your algorithm:
    * Now, you are ready to try your new algorithm! Simply run `goodtg`, *e.g.*, `goodtg --config_path configs/GOOD_configs/length/covariate/my_algorithm.yaml --gpu_idx 0`.
    * Alternatively, `python -m GOOD.kernel.main --config_path configs/GOOD_configs/length/covariate/my_algorithm.yaml --gpu_idx 0`.
5. Check your log:
    * After running, you can check your downloaded datasets, checkpoints, and log files in `storage` (defined in GOOD/definitions.py).
    * If you want to check various default storage setting, you may refer to :func:`process_configs <GOOD.utils.config_reader.process_configs>`.

:ref:`More questions <QA:Q&A>`.
