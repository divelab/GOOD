import time

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.train import train
from GOOD.networks.model_manager import load_model, config_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.args import args_parser
from GOOD.utils.initial import init
from GOOD.utils.logger import load_logger
from GOOD.kernel.evaluation import evaluate


def initialize_model_dataset(config: Union[CommonArgs, Munch]):
    # Initial
    init(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader


def load_task(task, model, loader, ood_algorithm, config: Union[CommonArgs, Munch], **kwargs):
    if task == 'train':
        train(model, loader, ood_algorithm, config)

    elif task == 'test':

        # config model
        print('#D#Config model and output the best checkpoint info...')
        test_score, test_loss = config_model(model, 'test', config=config)


def main():
    args = args_parser()
    config = config_summoner(args)
    load_logger(config)

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    load_task(config.task, model, loader, ood_algorithm, config)

    if config.task == 'train':
        load_task('test', model, loader, ood_algorithm, config)


if __name__ == '__main__':
    main()
