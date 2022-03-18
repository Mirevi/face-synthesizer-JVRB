from torch import nn, randn

from models.networks.blocks import ResnetBlock
from models.networks.traceable_network import TraceableNetwork


class ResidualMappingNetwork(TraceableNetwork):
    def __init__(self, input_nc, nmf=64, num_res_blocks=2, norm_layer=nn.BatchNorm2d):
        super(ResidualMappingNetwork, self).__init__()
        self.input_nc = input_nc
        use_dropout = True
        use_bias = True

        self.net = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, nmf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(nmf),
                    nn.ReLU(True)]

        for i in range(num_res_blocks):
            self.net += [ResnetBlock(nmf, 'zero', norm_layer, use_dropout, use_bias)]

        self.net += [nn.ReflectionPad2d(3)]
        self.net += [nn.Conv2d(nmf, nmf, kernel_size=7, padding=0)]
        self.net += [nn.Tanh()]

        self.net = nn.Sequential(*self.net)

    def input_noise(self, metadata):
        if type(metadata["image_size"]) is tuple:
            return randn((1, self.input_nc, metadata["image_size"][0], metadata["image_size"][1])).to(
                metadata["device"])
        else:
            return randn((1, self.input_nc, metadata["image_size"], metadata["image_size"])).to(metadata["device"])

    def forward(self, feature):
        return self.net(feature)
