class Register(object):

    def __init__(self):
        self.models = dict()
        self.datasets = dict()
        self.ood_algs = dict()

    def model_register(self, model_class):
        self.models[model_class.__name__] = model_class
        return model_class

    def dataset_register(self, dataset_class):
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    def ood_alg_register(self, ood_alg_class):
        self.ood_algs[ood_alg_class.__name__] = ood_alg_class
        return ood_alg_class


register = Register()
