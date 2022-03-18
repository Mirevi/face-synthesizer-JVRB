from enum import Enum

from config import BaseConfig, ConfigOptionPackage, ConfigOptionMetadata
from .multiscale_discriminator import MultiscaleDiscriminator
from .n_layer_discriminator import NLayerDiscriminator
from .pixel_discriminator import PixelDiscriminator
from ..initialization import InitializationType, NetworkInitializationCOP, init_weights
from ..network_cops import NetworkCOP
from ..normalization_layer import NormalizationType, NormalizationLayerCOP, get_norm_layer


class DiscriminatorCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(DiscriminatorType, 'netD', DiscriminatorType.n_layers,
                                 'specify discriminator architecture. The basic model is a 70x70 PatchGAN. '
                                 'n_layers allows you to specify the layers in the discriminator',
                                 choices=list(DiscriminatorType)),
            ConfigOptionMetadata(int, 'ndf', 64, '# of discrim filters in the first conv layer'),
            ConfigOptionMetadata(bool, 'accessible_intermediate_feat', False,
                                 'if intermediate features of discriminator are accessible', is_constant=True),
        ]

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        if options.netD is DiscriminatorType.multiscale:
            metadata.extend([ConfigOptionMetadata(int, 'num_D', 3, 'number of discriminators')])
        if options.netD is DiscriminatorType.n_layers or options.netD is DiscriminatorType.multiscale:
            metadata.extend([ConfigOptionMetadata(int, 'n_layers_D', 4, 'layers in the n_layer Discriminator.')])
        return metadata


class DiscriminatorType(Enum):
    multiscale = 'multiscale'
    n_layers = 'n_layers'
    pixel = 'pixel'

    def __str__(self):
        return self.value


def define_discriminator_from_config(config: BaseConfig):
    """Creates a discriminator with values from a config"""
    # validate config
    if NormalizationLayerCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if NetworkInitializationCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if NetworkCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if DiscriminatorCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    return define_discriminator(
        config['input_nc'] + config['output_nc'],  # conditional discriminator
        config['ndf'],
        config['netD'],
        config['norm_type'],
        config['init_type'],
        config['init_gain'],
        config['num_D'] if 'num_D' in config else None,
        config['n_layers_D'] if 'n_layers_D' in config else None,
        config['accessible_intermediate_feat'],
    )


def define_discriminator(input_nc, ndf, netD: DiscriminatorType, norm_type: NormalizationType = NormalizationType.batch,
                         init_type: InitializationType = InitializationType.normal, init_gain=0.02,
                         num_D=3, n_layers_D=3, accessible_intermediate_feat=False):
    """Create a discriminator"""
    norm_layer = get_norm_layer(norm_type)

    if netD == DiscriminatorType.multiscale:
        net = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, num_D, accessible_intermediate_feat)
    elif netD == DiscriminatorType.n_layers:
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer, accessible_intermediate_feat)
    elif netD == DiscriminatorType.pixel:
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model type {} is not implemented yet.'.format(netD))

    init_weights(net, init_type, init_gain)

    return net
