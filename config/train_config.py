import os

import sys

from config import COPWithModifiableDefaults, ConfigPackageProvider, StandardConfig, ArgparseConfig, \
    ConfigOptionMetadata, NameCOP, ContinueCOP, OverwriteCOP, ConfigDefaultModification, SaveDataCOP, LoadDataCOP
from data.datasets import DatasetTypeCOP, dataset_class
from data.datasets.base_dataset import DataLoaderCOP
from models import ModelTypeCOP, model_class, ModelType
from documentation import Documentation
from models.base_model import BaseModelCOP


class TrainCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'no_eval', False, 'no evaluation'),
            ConfigOptionMetadata(int, 'eval_epoch_freq', 10, 'frequency of evaluating the model'),
            ConfigOptionMetadata(int, 'save_epoch_freq', 2, 'frequency of saving checkpoints at the end of epochs'),
            ConfigOptionMetadata(int, 'continue_epoch', sys.maxsize,
                                 'Tries to load the checkpoint from the epoch or if not existing the latest below this '
                                 'epoch and continues the training process from the loaded checkpoint epoch.'),
            ConfigOptionMetadata(int, 'n_epochs', 100, 'number of epochs with the initial learning rate'),
            ConfigOptionMetadata(int, 'n_epochs_decay', 100,
                                 'number of epochs to linearly decay learning rate to zero'),
        ]

    @staticmethod
    def get_default_modifications() -> list:
        output_root = os.path.join(os.path.abspath(os.curdir), 'checkpoints')
        return [
            ConfigDefaultModification(BaseModelCOP, 'with_train_options', True),
            ConfigDefaultModification(SaveDataCOP, 'output_root', output_root),
            ConfigDefaultModification(ModelTypeCOP, 'model_type', ModelType.pix2pix_extended),
        ]


class TrainPackageProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        return [TrainCOP, DatasetTypeCOP, DataLoaderCOP, ModelTypeCOP, NameCOP, ContinueCOP, OverwriteCOP, SaveDataCOP]

    @staticmethod
    def get_required_providers() -> list:
        return [Documentation]

    @staticmethod
    def get_conditional_providers(options) -> list:
        return [
            dataset_class(options.dataset_type),
            model_class(options.model_type),
        ]


class TrainStandardConfig(StandardConfig):
    @staticmethod
    def from_string(s: str):
        arg_dict = StandardConfig.string_to_dict(s)
        return TrainStandardConfig(arg_dict)

    def __init__(self, args: dict = None):
        super().__init__(args)
        self.add_package_provider(TrainPackageProvider)


class TrainArgparseConfig(ArgparseConfig):
    def __init__(self, parser_args):
        super().__init__(parser_args)
        self.add_package_provider(TrainPackageProvider)


if __name__ == '__main__':
    config = TrainStandardConfig()
    config['name'] = 'temp'
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))

    arguments = ['--name', 'temp']
    arguments = None
    config = TrainArgparseConfig(arguments)
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))
