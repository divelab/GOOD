Customize Datasets, GNNs, OOD Algorithms
==========================================

Custom datasets
----------------

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
-------------

For customizing your GNN model, it is a good idea to inherit :class:`GNNBasic <GOOD.networks.models.BaseGNN.GNNBasic>` for flexible
argument reading.

*Register your GNN:*

.. code-block:: python

   from GOOD import register

   @register.model_register
   class MyGNN(GNNBasic):
       pass

Custom OOD algorithms
-----------------------

Current OOD algorithms can be implemented by overwriting :func:`input_preprocess <GOOD.ood_algorithms.algorithms.BaseOOD.BaseOODAlg.input_preprocess>`,
:func:`output_postprocess <GOOD.ood_algorithms.algorithms.BaseOOD.BaseOODAlg.output_postprocess>`,
:func:`input_preprocess <GOOD.ood_algorithms.algorithms.BaseOOD.BaseOODAlg.loss_calculate>`, and
:func:`loss_postprocess <GOOD.ood_algorithms.algorithms.BaseOOD.BaseOODAlg.loss_postprocess>`. With the help of passing
the algorithm object into your GNN, these processes covered the whole batched training pipeline.

Take a graph prediction task as an example.

.. code-block:: python

   # data augmentation algorithm may implement input_preprocess.
   data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training,
                                                                    config)

   # algorithms for changing model structure may implement custom GNNs and output_postprocess. Refer to DANN.
   model_output = model(data=data, ood_algorithm=ood_algorithm)
   # generally for multiple model outputs.
   raw_pred = ood_algorithm.output_postprocess(model_output)

   # calculate loss with reduction.
   loss = ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, config)
   # aggregate loss.
   loss = ood_algorithm.loss_postprocess(loss, data, mask, config)

*register your OOD algorithm:*

.. code-block:: python

   from GOOD import register

   @register.ood_alg_register
   class MyOODAlgorithm(BaseOODAlg):
       pass

For concrete details: :func:`train batch <GOOD.kernel.train.train_batch>`.

