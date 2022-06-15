
def test_import():
    from GOOD.ood_algorithms.algorithms.VREx import VREx
    from GOOD.ood_algorithms.algorithms.Mixup import Mixup
    from GOOD.ood_algorithms.algorithms.IRM import IRM
    from GOOD.ood_algorithms.algorithms.ERM import ERM
    from GOOD.ood_algorithms.algorithms.GroupDRO import GroupDRO
    from GOOD.ood_algorithms.algorithms.Coral import Coral
    from GOOD.ood_algorithms.algorithms.DANN import DANN

    from GOOD import register
    algs = ['VREx', 'Mixup', 'IRM', 'ERM', 'GroupDRO', 'Coral', 'DANN']
    for alg in algs:
        assert register.ood_algs[alg] is eval(alg)