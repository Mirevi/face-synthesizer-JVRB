import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

# All data parameters import
from .backbone_nets.mobilenetv2_backbone import mobilenet_v2
from .utils.params import ParamsPack

param_pack = ParamsPack()


# Image-to-parameter
class I2P(nn.Module):
    def __init__(self):
        super(I2P, self).__init__()
        self.backbone = mobilenet_v2(pretrained=False)

    def forward_test(self, input):
        """ Testing time forward."""
        _3D_attr, avgpool = self.backbone(input)
        return _3D_attr, avgpool


# Main model SynergyNet definition
class SynergyNet(nn.Module):
    def __init__(self):
        super(SynergyNet, self).__init__()
        # Image-to-parameter
        self.I2P = I2P()
        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_fp = os.path.join('', sys.modules[__name__].__package__.replace('.', '\\'), 'pretrained',
                                     'best.pth.tar')
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']

        model_dict = self.state_dict()

        # because the model is trained by multiple gpus, prefix 'module' should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]

        self.load_state_dict(OrderedDict(model_dict), strict=False)

    def forward_test(self, input):
        """test time forward"""
        _3D_attr, _ = self.I2P.forward_test(input)
        return _3D_attr


if __name__ == '__main__':
    pass
