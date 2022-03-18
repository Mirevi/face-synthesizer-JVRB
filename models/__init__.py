"""This package contains modules related to objective functions, optimizations, and network architectures.

See the l1_model.py file for an example of a model class.
To add a custom model follow the example of the l1_model and add an entry in the ModelType Enum.
"""

import importlib
import sys
from enum import Enum

from config import ConfigOptionPackage, ConfigOptionMetadata
from models.base_model import BaseModel, ModelMode


class ModelType(Enum):
    """
    All implemented Model Types.
    """
    l1 = "l1"
    pix2pix = 'pix2pix'
    pix2pix_extended = 'pix2pix_extended'

    def __str__(self):
        return self.value


class ModelTypeCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(ModelType, 'model_type', ModelType.l1, 'chooses which model to use.',
                                 choices=list(ModelType)),
        ]


def model_class(model_type: ModelType) -> BaseModel.__class__:
    """
    Imports the model module and returns the class of the model.
    """
    name = str(model_type)
    module_filename = "." + name.lower() + "_model"
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    model_class = None

    model_class_name = name.replace('_', '') + 'model'
    for name, cls in module.__dict__.items():
        if name.lower() == model_class_name.lower() and issubclass(cls, BaseModel):
            model_class = cls

    if model_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            module_filename, model_class_name))

    return model_class


def new_model_instance(config, initial_model_mode: ModelMode, model_type: ModelType = None) -> BaseModel:
    if model_type is not None:
        model_type = model_type
    elif ModelTypeCOP in config:
        model_type = config['model_type']
    else:
        raise RuntimeError('Cannot determine which model type to instantiate. '
                           'Either add ModelTypeCOP to config and specify model_type or specify model_type parameter.')

    type_class = model_class(model_type)
    instance = type_class(config, initial_model_mode)
    print("model [%s] was created" % type(instance).__name__)
    return instance
