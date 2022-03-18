import os
from argparse import ArgumentParser

from config import ConfigDefaultModification, COPWithModifiableDefaults, ConfigOptionMetadata, StandardConfig, \
    ArgparseConfig, ConfigPackageProvider
from config.config_option_packages import DepthMaskCOP, LoadDataCOP
from data.datasets import DatasetTypeCOP
from data.datasets.base_dataset import DatasetCOP, BaseDataset, dataset_class


class EvaluateCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'opt_file_name', 'TrainConfig.txt', '"The name of the opt files. All opt files must have this name.'),
            ConfigOptionMetadata(bool, 'uniform_eval_dataset', False, 'uses the specified eval dataset settings for all evaluations instead of the value specified in the opt files.'),
            ConfigOptionMetadata(bool, 'uniform_depth_mask_usage', False, 'uses the specified no_depth_mask value for all models instead of the value specified in the opt files.'),
        ]

    @staticmethod
    def get_default_modifications() -> list:
        input_root = os.path.join(os.path.abspath(os.curdir), 'checkpoints')
        return [
            ConfigDefaultModification(LoadDataCOP, 'input_root', input_root),
        ]


class EvaluatePackageProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        DatasetTypeCOP.use_mode = DatasetTypeCOP.UseModes.Eval
        return [
            EvaluateCOP, DatasetTypeCOP, LoadDataCOP
        ]

    @staticmethod
    def get_conditional_option_packages(options) -> list:
        packages = []
        if options.uniform_depth_mask_usage:
            packages.extend([DepthMaskCOP])
        return packages

    @staticmethod
    def get_conditional_providers(options) -> list:
        providers = []
        if options.uniform_eval_dataset:
            providers.append(dataset_class(options.dataset_type))
        return providers


class EvaluateStandardConfig(StandardConfig):
    def __init__(self, args: dict = None):
        super().__init__(args)
        self.add_package_provider(EvaluatePackageProvider)


class EvaluateArgparseConfig(ArgparseConfig):
    def __init__(self, parser_args=None):
        super().__init__(parser_args)
        self.add_package_provider(EvaluatePackageProvider)


if __name__ == '__main__':
    arguments = {'uniform_depth_mask_usage': True}
    config = EvaluateStandardConfig(arguments)
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))

    arguments = ['--uniform_eval_dataset']
    config = EvaluateArgparseConfig(arguments)
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))
