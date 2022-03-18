import importlib
import sys
from enum import Enum
from config.base_config import ConfigOptionMetadata, ConfigDefaultModification, ConfigOptionPackage, \
    COPWithModifiableDefaults, ConfigPackageProvider, BaseConfig, StandardConfig, ArgparseConfig
from config.config_option_packages import *


class ConfigType(Enum):
    Capture = 'Capture'
    Convert = 'Convert'
    Train = 'Train'
    Test = 'Test'
    Evaluate = 'Evaluate'

    def __str__(self):
        return self.value


def config_class(config_type: ConfigType, subclass) -> BaseConfig.__class__:
    config_type = str(config_type)
    module_filename = '.' + config_type.lower() + '_config'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    type_class = None

    type_class_name = config_type.replace('_', '') + subclass.__name__.replace('_', '')
    for name, cls in module.__dict__.items():
        if name.lower() == type_class_name.lower() and issubclass(cls, subclass):
            type_class = cls

    if type_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of %s with class name that matches %s in lowercase." % (
                module_filename, subclass.__name__, type_class_name))
    return type_class


def new_default_config(config_type: ConfigType) -> StandardConfig:
    type_class = config_class(config_type, StandardConfig)
    return type_class()


def new_argparse_config(config_type: ConfigType, parser_args=None) -> ArgparseConfig:
    type_class = config_class(config_type, ArgparseConfig)
    return type_class(parser_args)
