Installation
==============

Conda dependencies
--------------------

GOOD depends on `PyTorch (>=1.6.0) <https://pytorch.org/get-started/previous-versions/>`_, `PyG (>=2.0) <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_, and
`RDKit (>=2020.09.5) <https://www.rdkit.org/docs/Install.html>`_. For more details: `conda environment <https://github.com/divelab/GOOD/blob/docs/environment.yml>`_

.. note::
   Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.


GOOD installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   git clone https://github.com/divelab/GOOD.git && cd GOOD
   # In your anaconda environment
   pip install -e .
