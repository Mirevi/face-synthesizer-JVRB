from torch import nn, cat

from .mlp_conv_mapping_network import MLPConvMappingNetwork
from .residual_mapping_network import ResidualMappingNetwork
from ..traceable_network import TraceableNetwork


class MLPAndResidualMappingNetwork(TraceableNetwork):
    def __init__(self, input_nc, num_landmark_vars, nmf=64, num_upsamples=5, num_res_blocks=2,
                 norm_layer=nn.BatchNorm2d):
        super(MLPAndResidualMappingNetwork, self).__init__()
        nmf_extra = nmf % 2
        self.mlp = MLPConvMappingNetwork(num_landmark_vars, int(nmf / 2) + nmf_extra, num_upsamples, norm_layer)
        self.residual = ResidualMappingNetwork(input_nc, int(nmf / 2), num_res_blocks, norm_layer)

    def input_noise(self, metadata):
        return {'landmarks': self.mlp.input_noise(metadata), 'feature': self.residual.input_noise(metadata)}

    def forward(self, input):
        return cat((self.mlp(input['landmarks']), self.residual(input['feature'])), 1)
