.. GOOD documentation master file

GOOD documentation
========================================

**GOOD** (Graph OOD) is a graph out-of-distribution (OOD) algorithm benchmarking library depending on PyTorch and PyG
to make develop and benchmark OOD algorithms easily.

GOOD provides a suite of dataset, GNN, OOD algorithm APIs and constructs an easy-to-use OOD algorithm training and testing bed.
All :mod:`algorithms <GOOD.ood_algorithms.algorithms>`, :mod:`GNNs <GOOD.networks.models>`, and :mod:`datasets <GOOD.data.good_datasets>`
can be easily registered by :obj:`register <GOOD.utils.register>` and automatically embedded into the designed pipeline like a breeze! Using this register
and overwriting classes, you can custom your methods and embed them into the test bed conveniently.

Currently, GOOD contains 11 datasets with 17 domain selections. When combined with covariate, concept, and no shifts, we obtain 51 different splits.
We provide performance results on 10 commonly used baseline methods (ERM, IRM, VREx, GroupDRO, Coral, DANN, Mixup, DIR, EERM, SRGNN).
This results in 510 dataset-model combinations in total. Our results show significant performance gaps between in-distribution and OOD settings.
This GOOD benchmark is a growing project and expects to expand in quantity and variety of resources as the area develops.
Any contribution is welcomed!


.. image:: imgs/Datasets.png
   :width: 680
   :alt: GOOD datasets

----------------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Notes

   installation
   tutorial
   configs
   custom
   leaderboard
   contributing
   QA


Package reference
-------------------

.. autosummary::
   :toctree: _autosummary
   :caption: Package reference
   :recursive:

   GOOD.data
   GOOD.definitions
   GOOD.kernel
   GOOD.networks
   GOOD.ood_algorithms
   GOOD.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
