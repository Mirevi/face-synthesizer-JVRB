from enum import Enum

from config import BaseConfig, ConfigOptionPackage, ConfigOptionMetadata
from .mlp_and_residual_mapping_network import MLPAndResidualMappingNetwork
from .residual_mapping_network import ResidualMappingNetwork
from .mlp_conv_mapping_network import MLPConvMappingNetwork
from ..initialization import InitializationType, init_weights, NetworkInitializationCOP
from ..normalization_layer import NormalizationType, get_norm_layer, NormalizationLayerCOP


class MappingNetworkCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(MappingNetworkType, 'netM', MappingNetworkType.mlp_and_residual_512,
                                 'specify mapping_network network architecture [mlp_with_conv, residual, '
                                 'mlp_and_residual_512]', choices=list(MappingNetworkType)),
            ConfigOptionMetadata(int, 'nmf', 64, '# of mapping_network filters in the last conv layer'),
            ConfigOptionMetadata(int, 'num_landmarks', 70, 'number of landmarks'),
            ConfigOptionMetadata(int, 'mapping_res_blocks', 3, '# of residual mapping_network blocks in the Mapping Network'),
            ConfigOptionMetadata(bool, 'no_mapping_feat_concatenation', False, 'No concatenation of the result of the mapping_network network with the real feature image as input for the Generator.'),
        ]


class MappingNetworkType(Enum):
    mlp_with_conv_512 = 'mlp_with_conv_512'
    residual = 'residual'
    mlp_and_residual_512 = 'mlp_and_residual_512'

    def __str__(self):
        return self.value


def define_mapping_network_from_config(config: BaseConfig):
    # validate config
    if NormalizationLayerCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if NetworkInitializationCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')
    if MappingNetworkCOP not in config:
        raise RuntimeError('Necessary Package is not in current Config!')

    return define_mapping_network(
        config['netM'],
        config['input_nc'],
        config['num_landmarks'] * 2,
        config['mapping_res_blocks'],
        config['nmf'],
        config['norm_type'],
        config['init_type'],
        config['init_gain'],
    )


def define_mapping_network(netM: MappingNetworkType, input_nc, num_landmark_vars=140, num_res_blocks=2, nmf=64,
                           norm_type: NormalizationType = NormalizationType.batch,
                           init_type: InitializationType = InitializationType.normal, init_gain=0.02):
    norm_layer = get_norm_layer(norm_type)

    if netM == MappingNetworkType.mlp_with_conv_512:
        net = MLPConvMappingNetwork(num_landmark_vars, nmf, 5, norm_layer)
    elif netM == MappingNetworkType.residual:
        net = ResidualMappingNetwork(input_nc, nmf, num_res_blocks, norm_layer)
    elif netM == MappingNetworkType.mlp_and_residual_512:
        net = MLPAndResidualMappingNetwork(input_nc, num_landmark_vars, nmf, 5, num_res_blocks, norm_layer)
    else:
        raise NotImplementedError('Mapping network model type {} is not implemented yet.'.format(netM))

    init_weights(net, init_type, init_gain)

    return net


def get_mapping_network_input(netM: MappingNetworkType, landmarks, feature):
    if netM == MappingNetworkType.mlp_with_conv_512:
        return landmarks
    elif netM == MappingNetworkType.residual:
        return feature
    elif netM == MappingNetworkType.mlp_and_residual_512:
        return {'landmarks': landmarks, 'feature': feature}
    else:
        raise NotImplementedError('Mapping network model type {} is not implemented yet.'.format(netM))
