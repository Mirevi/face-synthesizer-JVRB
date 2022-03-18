import importlib
from abc import ABC

import sys
from enum import Enum

from config import ConfigOptionPackage, ConfigOptionMetadata, ConfigPackageProvider


class DepthFillingAlgorithm(Enum):
    FDCBOP = 'FDCBOP'

    def __str__(self):
        return self.value


class DepthFillingCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'no_depth_filling', False, 'Do not fill holes in depth images.'),
            ConfigOptionMetadata(DepthFillingAlgorithm, 'df_algorithm', DepthFillingAlgorithm.FDCBOP, 'The algorithm used for depth hole filling', choices=list(DepthFillingAlgorithm)),
        ]


class DepthFillerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'df_padding', 150, 'Padding for depth filling process. Can helpt to fill holes because more data is available for the process.'),
            ConfigOptionMetadata(int, 'df_image_size', 400, 'Image size of the images used for the depth hole filling process. Changing this value may affect performance.'),
        ]


class DepthFiller(ConfigPackageProvider, ABC):
    def __init__(self, config):
        # validate config
        if self not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

    @staticmethod
    def get_required_option_packages() -> list:
        return [DepthFillerCOP]


def depth_filler_class(df_algorithm_type: DepthFillingAlgorithm):
    name = str(df_algorithm_type)
    module_filename = '.' + name.lower() + '_depth_filler'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    algorithm_class = None

    algorithm_class_name = name.replace('_', '') + 'depthfiller'
    for name, cls in module.__dict__.items():
        if name.lower() == algorithm_class_name.lower() and issubclass(cls, DepthFiller):
            algorithm_class = cls

    if algorithm_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of DepthFiller with class name that matches %s in lowercase." % (
                module_filename, algorithm_class_name))
    return algorithm_class


def new_depth_filler_instance(df_algorithm: DepthFillingAlgorithm, opt):
    algorithm_class = depth_filler_class(df_algorithm)
    return algorithm_class(opt)

