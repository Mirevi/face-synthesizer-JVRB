from enum import Enum

import torch
from torch import nn

from config import ConfigOptionPackage, ConfigOptionMetadata


class GANLossCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(GANMode, 'gan_mode', GANMode.lsgan,
                                 'the type of GAN objective. vanilla GAN loss is the cross-entropy objective used in '
                                 'the original GAN paper.',
                                 choices=list(GANMode)),
        ]


class GANMode(Enum):
    vanilla = 'vanilla'
    lsgan = 'lsgan'

    def __str__(self):
        return self.value


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode is GANMode.lsgan:
            self.loss = nn.MSELoss()
        elif gan_mode is GANMode.vanilla:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("gan mode '{}' not implemented".format(gan_mode))

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if isinstance(prediction[0], list):
            loss = 0
            for prediction_i in prediction:
                pred = prediction_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        elif isinstance(prediction, list):
            target_tensor = self.get_target_tensor(prediction[-1], target_is_real)
            return self.loss(prediction[-1], target_tensor)
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)
