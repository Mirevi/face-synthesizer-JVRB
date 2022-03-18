import functools
from enum import Enum

from torch import nn

from config import ConfigOptionPackage, ConfigOptionMetadata, BaseConfig


class NormalizationLayerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(NormalizationType, 'norm_type', NormalizationType.instance,
                                 'normalization layer type', choices=list(NormalizationType)),
        ]


class Identity(nn.Module):
    def forward(self, x):
        return x


class NormalizationType(Enum):
    batch = 'batch'
    instance = 'instance'
    none = 'none'

    def __str__(self):
        return self.value


def get_norm_layer_from_config(config: BaseConfig):
    """Returns a normalization layer which is specified in a config."""
    # validate config
    if NormalizationLayerCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    return get_norm_layer(config['norm_type'])


def get_norm_layer(norm_type):
    """Returns a normalization layer"""
    if norm_type == NormalizationType.batch:
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == NormalizationType.instance:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == NormalizationType.none:
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer type {} is not implemented yet'.format(norm_type))
    return norm_layer
