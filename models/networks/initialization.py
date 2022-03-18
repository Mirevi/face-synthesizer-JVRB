from enum import Enum

from torch.nn import init

from config import ConfigOptionPackage, ConfigOptionMetadata


class NetworkInitializationCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(InitializationType, 'init_type', InitializationType.normal,
                                 'network initialization type'),
            ConfigOptionMetadata(float, 'init_gain', 0.02, 'scaling factor for some init.'),
        ]


class InitializationType(Enum):
    normal = 'normal'
    xavier = 'xavier'
    kaiming = 'kaiming'
    orthogonal = 'orthogonal'

    def __str__(self):
        return self.value


def init_weights_with_config_values(config, net):
    """Initialize network weights with values from config."""
    # validate config
    if NetworkInitializationCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    init_weights(net, config['init_type'], config['init_gain'])


def init_weights(net, init_type: InitializationType = InitializationType.normal, init_gain: float = 0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (InitType) -- the initialization method
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == InitializationType.normal:
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == InitializationType.xavier:
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == InitializationType.kaiming:
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == InitializationType.orthogonal:
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization type {} is not implemented yet'.format(init_type))

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
