import torch
from torchsummary import summary

from config import COPWithModifiableDefaults, ConfigOptionMetadata, ConfigDefaultModification
from metrics import MetricType, new_metric_instance, metric_class
from . import networks, BaseModel, ModelMode
from .base_model import AdamOptimizerCOP
from .networks import GANMode, GeneratorType, NormalizationType, GANLossCOP, ConvGANPackageProvider, \
    define_generator_from_config, define_discriminator_from_config, GeneratorCOP, NormalizationLayerCOP, get_scheduler


class Pix2PixModelCOP(COPWithModifiableDefaults):
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


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping_network from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(Pix2PixModel, Pix2PixModel).get_required_option_packages()
        packages.extend([Pix2PixModelCOP, AdamOptimizerCOP])
        return packages

    @staticmethod
    def get_required_providers() -> list:
        providers = super(Pix2PixModel, Pix2PixModel).get_required_providers()
        providers.extend([
            ConvGANPackageProvider,
            metric_class(MetricType.Pixel_Accuracy),
            metric_class(MetricType.Threshold_Pixel_Accuracy),
            metric_class(MetricType.PSNR),
            metric_class(MetricType.SSIM),
            metric_class(MetricType.FID),
            metric_class(MetricType.LPIPS),
        ])
        return providers

    def __init__(self, config, initial_model_mode=ModelMode.Inference):
        """Initialize the pix2pix class."""
        BaseModel.__init__(self, config, initial_model_mode)
        # get necessary values from config
        self.input_nc = config['input_nc']
        self.output_nc = config['output_nc']
        self.use_depth_mask = not config['no_depth_mask']

        # initialize members
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_feature', 'real_color', 'fake_color', 'real_depth', 'fake_depth']
        self.model_names = ['G']
        self.trace_model_names = ['G']
        self.metric_names.extend(['pixel_accuracy', 'threshold_pixel_accuracy', 'psnr', 'ssim', 'fid', 'lpips'])

        # define generator
        self.netG = define_generator_from_config(config)
        self.netG.to(self.device)

        if self.initialize_complete_model:
            # get necessary values from config
            self.lambda_L1 = config['lambda_L1']

            # define a discriminator as cGAN
            self.netD = define_discriminator_from_config(config)
            self.netD.to(self.device)
            self.discriminator_networks = [self.netD]
            self.model_names.append('D')

            # define loss functions
            self.criterionGAN = networks.GANLoss(config['gan_mode']).to(self.device)
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
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
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
        self.loss_D_real = None
        self.loss_D_fake = None
        self.loss_D = None

    def print_summary(self, image_size: tuple):
        print("Summarize_net_G")
        summary(self.netG, (self.input_nc, image_size[0], image_size[1]))
        print("Summarize_net_D")
        summary(self.netD, (self.input_nc + self.output_nc, image_size[0], image_size[1]))

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
        """Run forward pass"""
        self.fake_color_depth = self.netG(self.real_feature)  # G(A)

        # apply depth mask if necessary
        if self.use_depth_mask:
            self.fake_color_depth = self.apply_depth_as_mask(self.fake_color_depth)

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
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

    def optimize_parameters(self):
        BaseModel.optimize_parameters(self)
        self.forward()  # compute fake images: G(A)

        # update D
        self.set_requires_grad(self.discriminator_networks, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights

        # update G
        self.set_requires_grad(self.discriminator_networks, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update G's weights

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # we use conditional GANs; we need to feed both input and output to the discriminator
        input_D_real = torch.cat((self.real_feature, self.real_color_depth), 1)
        input_D_fake = torch.cat((self.real_feature, self.fake_color_depth), 1)

        self.loss_D, self.loss_D_real, self.loss_D_fake = self.discriminator_losses(self.netD, input_D_real, input_D_fake)
        self.loss_D.backward()

    def discriminator_losses(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real input
            fake (tensor array) -- fake input (with data from generator)

        Return losses: 1.loss_D, 2.loss_D_real, 3.loss_D_fake.
        """
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D, loss_D_real, loss_D_fake

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        input_D = torch.cat((self.real_feature, self.fake_color_depth), 1)
        self.pred_D_fake = self.netD(input_D)
        self.loss_G_GAN = self.criterionGAN(self.pred_D_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_color_depth, self.real_color_depth) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()