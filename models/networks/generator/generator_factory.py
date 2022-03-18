from enum import Enum

from config import BaseConfig, ConfigOptionPackage, ConfigOptionMetadata
from .resnet_generator import ResnetGenerator
from .unet_generator import UnetGenerator
from ..initialization import InitializationType, init_weights, NetworkInitializationCOP
from ..network_cops import NetworkCOP
from ..normalization_layer import NormalizationType, get_norm_layer, NormalizationLayerCOP


class GeneratorCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(GeneratorType, 'netG', GeneratorType.unet_512, 'specify generator architecture',
                                 choices=list(GeneratorType)),
            ConfigOptionMetadata(int, 'ngf', 64, '# of gen filters in the last conv layer'),
            ConfigOptionMetadata(bool, 'no_dropout', False, 'no dropout for the generator'),
        ]


class GeneratorType(Enum):
    resnet_9blocks = 'resnet_9blocks'
    resnet_6blocks = 'resnet_6blocks'
    unet_128 = 'unet_128'
    unet_256 = 'unet_256'
    unet_512 = 'unet_512'

    def __str__(self):
        return self.value


def define_generator_from_config(config: BaseConfig):
    # validate config
    if NormalizationLayerCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if NetworkInitializationCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if NetworkCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if GeneratorCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    return define_generator(
        config['input_nc'],
        config['output_nc'],
        config['ngf'],
        config['netG'],
        config['norm_type'],
        not config['no_dropout'],
        config['init_type'],
        config['init_gain'],
    )


def define_generator(input_nc, output_nc, ngf, netG: GeneratorType,
                     norm_type: NormalizationType = NormalizationType.batch, use_dropout=False,
                     init_type: InitializationType = InitializationType.normal, init_gain=0.02):
    """Create a generator"""
    norm_layer = get_norm_layer(norm_type)

    if netG == GeneratorType.resnet_9blocks:
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == GeneratorType.resnet_6blocks:
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == GeneratorType.unet_128:
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == GeneratorType.unet_256:
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == GeneratorType.unet_512:
        net = UnetGenerator(input_nc, output_nc, 9, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model type {} is not implemented yet.'.format(netG))

    init_weights(net, init_type, init_gain)

    return net
