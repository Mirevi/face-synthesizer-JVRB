from argparse import ArgumentParser

import torch
from torchsummary import summary

from config import COPWithModifiableDefaults, ConfigOptionMetadata, ConfigDefaultModification, DepthMaskCOP
from metrics import MetricType, new_metric_instance, metric_class
from . import BaseModel
from .base_model import AdamOptimizerCOP, ModelMode
from .networks import NormalizationLayerCOP, GeneratorCOP, NormalizationType, GeneratorType, \
    NetworkBasePackageProvider, define_generator_from_config, get_scheduler


class L1ModelCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        if options.with_train_options:
            metadata.extend([
                ConfigOptionMetadata(float, 'lambda_L1', 100.0, 'weight for L1 loss'),
            ])
        return metadata

    @staticmethod
    def get_default_modifications() -> list:
        return [
            ConfigDefaultModification(NormalizationLayerCOP, 'norm_type', NormalizationType.batch),
            ConfigDefaultModification(GeneratorCOP, 'netG', GeneratorType.unet_512),
        ]


class L1Model(BaseModel):
    """This model has a generator which is trained with the L1 loss only."""
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(L1Model, L1Model).get_required_option_packages()
        packages.extend([L1ModelCOP, AdamOptimizerCOP, DepthMaskCOP, GeneratorCOP])
        return packages

    @staticmethod
    def get_required_providers() -> list:
        providers = super(L1Model, L1Model).get_required_providers()
        providers.extend([
            NetworkBasePackageProvider,
            metric_class(MetricType.Pixel_Accuracy),
            metric_class(MetricType.Threshold_Pixel_Accuracy),
            metric_class(MetricType.PSNR),
            metric_class(MetricType.SSIM),
            metric_class(MetricType.FID),
            metric_class(MetricType.LPIPS),
        ])
        return providers

    def __init__(self, config, initial_model_mode: ModelMode):
        """Initialize the l1 class.

        Parameters:
            config (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, config, initial_model_mode)
        # get necessary values from config
        self.input_nc = config['input_nc']
        self.use_depth_mask = not config['no_depth_mask']

        # initialize members
        self.loss_names = ['G_L1']
        self.visual_names = ['real_feature', 'real_color', 'fake_color', 'real_depth', 'fake_depth']
        self.model_names = ['G']
        self.trace_model_names = ['G']
        self.metric_names.extend(['pixel_accuracy', 'threshold_pixel_accuracy', 'psnr', 'ssim', 'fid', 'lpips'])

        # define networks (both generator and discriminator)
        self.netG = define_generator_from_config(config)
        self.netG.to(self.device)

        # define and initialize necessary objects for training
        if self.initialize_complete_model:
            # get necessary values from config
            self.lambda_L1 = config['lambda_L1']

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()

            # define metrics
            self.metric_pixel_accuracy = new_metric_instance(MetricType.Pixel_Accuracy, config)
            self.metric_threshold_pixel_accuracy = new_metric_instance(MetricType.Threshold_Pixel_Accuracy, config)
            self.metric_psnr = new_metric_instance(MetricType.PSNR, config)
            self.metric_ssim = new_metric_instance(MetricType.SSIM, config)
            self.metric_fid = new_metric_instance(MetricType.FID, config)
            self.metric_lpips = new_metric_instance(MetricType.LPIPS, config)

        # initialize optimizers
        if self.initial_model_mode is ModelMode.Train:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
            self.optimizers = [self.optimizer_G]
            self.schedulers = [get_scheduler(config, optimizer, 0) for optimizer in self.optimizers]

        # declare necessary members
        self.real_feature = None
        self.real_color_depth = None
        self.real_landmarks = None
        self.fake_color_depth = None
        self.real_color = None
        self.real_depth = None
        self.fake_color = None
        self.fake_depth = None
        self.loss_G_L1 = None
        self.loss_G = None

    def print_summary(self, image_size: tuple):
        print("Summarize_net_G")
        summary(self.netG, (self.input_nc, image_size[0], image_size[1]))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_feature = input['Feature'].to(self.device)
        self.real_color_depth = input['ColorWithDepth'].to(self.device)
        self.real_landmarks = input['Landmarks'].to(self.device)

        # apply depth mask if necessary
        if self.use_depth_mask:
            self.real_color_depth = self.apply_depth_as_mask(self.real_color_depth)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_color_depth = self.netG(self.real_feature)  # G(A)

        # apply depth mask if necessary
        if self.use_depth_mask:
            self.fake_color_depth = self.apply_depth_as_mask(self.fake_color_depth)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        # compute color and depth visuals
        self.real_color = torch.index_select(self.real_color_depth, 1, torch.tensor([0, 1, 2]).to(self.device))
        self.real_depth = torch.index_select(self.real_color_depth, 1, torch.tensor([3]).to(self.device))
        self.fake_color = torch.index_select(self.fake_color_depth, 1, torch.tensor([0, 1, 2]).to(self.device))
        self.fake_depth = torch.index_select(self.fake_color_depth, 1, torch.tensor([3]).to(self.device))

    def evaluate(self):
        super().evaluate()

        self.metric_pixel_accuracy.add_image_pairs_to_statistics(self.real_color_depth, self.fake_color_depth)
        self.metric_threshold_pixel_accuracy.add_image_pairs_to_statistics(self.real_color_depth,
                                                                           self.fake_color_depth)
        self.metric_psnr.add_image_pairs_to_statistics(self.real_color_depth, self.fake_color_depth)
        self.metric_ssim.add_image_pairs_to_statistics(self.real_color_depth, self.fake_color_depth)
        self.metric_fid.add_image_pairs_to_statistics(self.real_color, self.fake_color)
        self.metric_lpips.add_image_pairs_to_statistics(self.real_color, self.fake_color)
        self.metric_lpips.add_image_pairs_to_statistics(self.real_depth, self.fake_depth)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_color_depth, self.real_color_depth) * self.lambda_L1

        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        BaseModel.optimize_parameters(self)
        self.forward()  # compute fake images: G(A)

        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
