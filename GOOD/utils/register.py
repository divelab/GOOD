r"""A kernel module that contains a global register for unified model, dataset, and OOD algorithms access.
"""

class Register(object):
    r"""
    Global register for unified model, dataset, and OOD algorithms access.
    """

    def __init__(self):
        self.pipelines = dict()
        self.launchers = dict()
        self.models = dict()
        self.datasets = dict()
        self.dataloader = dict()
        self.ood_algs = dict()

    def pipeline_register(self, pipeline_class):
        r"""
        Register for pipeline access.

        Args:
            pipeline_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.pipelines[pipeline_class.__name__] = pipeline_class
        return pipeline_class

    def launcher_register(self, launcher_class):
        r"""
        Register for pipeline access.

        Args:
            launcher_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.launchers[launcher_class.__name__] = launcher_class
        return launcher_class

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

    def dataloader_register(self, dataloader_class):
        r"""
        Register for dataloader access.

        Args:
            dataloader_class (class): dataloader class

        Returns (class):
            dataloader class

        """
        self.dataloader[dataloader_class.__name__] = dataloader_class
        return dataloader_class

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
