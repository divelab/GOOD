r"""A kernel module that contains a global register for unified model, dataset, and OOD algorithms access.
"""

class Register(object):
    r"""
    Global register for unified model, dataset, and OOD algorithms access.
    """

    def __init__(self):
        self.models = dict()
        self.datasets = dict()
        self.ood_algs = dict()

    def model_register(self, model_class):
        r"""
        Register for model access.

        Args:
            model_class (class): model class

        Returns (class):
            model class

        """
        self.models[model_class.__name__] = model_class
        return model_class

    def dataset_register(self, dataset_class):
        r"""
        Register for dataset access.

        Args:
            dataset_class (class): dataset class

        Returns (class):
            dataset class

        """
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    def ood_alg_register(self, ood_alg_class):
        r"""
        Register for OOD algorithms access.

        Args:
            ood_alg_class (class): OOD algorithms class

        Returns (class):
            OOD algorithms class

        """
        self.ood_algs[ood_alg_class.__name__] = ood_alg_class
        return ood_alg_class


register = Register()  #: The GOOD register object used for accessing models, datasets and OOD algorithms.
