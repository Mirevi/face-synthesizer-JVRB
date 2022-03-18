from argparse import ArgumentParser

import time

import itertools

import torch
from lpips import lpips
from torchsummary import summary

from config import ConfigOptionMetadata, COPWithModifiableDefaults, ConfigDefaultModification
from . import ModelMode
from .networks import DiscriminatorCOP, MappingNetworkCOP, define_mapping_network_from_config, define_generator, \
    DiscriminatorType, define_discriminator_from_config, get_scheduler, define_discriminator
from .networks.mapping_network.mapping_network_factory import get_mapping_network_input
from .pix2pix_model import Pix2PixModel


class Pix2PixExtendedModelCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'use_mapping_network', False, 'Uses a mapping_network network which maps the '
                                                                     'landmarks variables to several feature maps '
                                                                     'which are the input for the Generator.'),
        ]

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        if options.with_train_options:
            metadata.extend([
                ConfigOptionMetadata(bool, 'no_L1_loss', False, 'Does not use L1 loss.'),
                ConfigOptionMetadata(bool, 'use_gan_feat_loss', False, 'Uses p2phd feature loss as additional loss.'),
                ConfigOptionMetadata(float, 'lambda_gan_feat', 10.0, 'weight for p2phd feature loss'),
                ConfigOptionMetadata(bool, 'use_lpips_loss', False, 'Uses lpips as additional loss.'),
                ConfigOptionMetadata(float, 'lambda_lpips', 100.0, 'weight for lpips loss'),
                ConfigOptionMetadata(bool, 'use_cycle_loss', False, 'Uses cycle consistency loss as additional loss.'),
                ConfigOptionMetadata(float, 'lambda_cycle_forward', 10.0,
                                     'weight for cycle loss (Feature -> ColorDepth -> Feature)'),
                ConfigOptionMetadata(float, 'lambda_cycle_backward', 10.0,
                                     'weight for cycle loss (ColorDepth -> Feature -> ColorDepth)'),
            ])
        return metadata

    @staticmethod
    def get_conditional_default_modifications(options) -> list:
        modifications = []
        if options.with_train_options and options.use_gan_feat_loss:
            modifications.extend([ConfigDefaultModification(DiscriminatorCOP, 'accessible_intermediate_feat', True)])
        return modifications


class Pix2PixExtendedModel(Pix2PixModel):
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(Pix2PixExtendedModel, Pix2PixExtendedModel).get_required_option_packages()
        packages.append(Pix2PixExtendedModelCOP)
        return packages

    @staticmethod
    def get_conditional_option_packages(options) -> list:
        packages = super(Pix2PixExtendedModel, Pix2PixExtendedModel).get_conditional_option_packages(options)
        if options.use_mapping_network:
            packages.extend([MappingNetworkCOP])
        return packages

    def __init__(self, config, initial_model_mode=ModelMode.Inference):
        """Initialize the pix2pix class."""
        Pix2PixModel.__init__(self, config, initial_model_mode)
        # get necessary values from config
        self.use_mapping_network = config['use_mapping_network']

        # initial state
        # self.visual_names = ['real_feature', 'real_color', 'fake_color', 'real_depth', 'fake_depth']
        self.generator_networks = []
        self.discriminator_networks = []
        g_input_nc = self.input_nc

        # define M
        if self.use_mapping_network:
            self.netM_type = config['netM']
            self.nmf = config['nmf']
            self.use_mapping_feat_concatenation = not config['no_mapping_feat_concatenation']

            self.netM = define_mapping_network_from_config(config)
            self.netM.to(self.device)
            self.generator_networks.append(self.netM)
            self.model_names.extend(['M'])
            self.trace_model_names.extend(['M'])
            self.visual_names.extend(['real_mapped_feature_example'])

            if self.use_mapping_feat_concatenation:
                g_input_nc = self.nmf + 1
            else:
                g_input_nc = self.nmf

        # define G
        self.netG = define_generator(g_input_nc, config['output_nc'], config['ngf'], config['netG'],
                                     config['norm_type'], not config['no_dropout'], config['init_type'],
                                     config['init_gain'])
        self.netG.to(self.device)
        self.generator_networks.append(self.netG)

        if self.initialize_complete_model:
            # get necessary values from config
            self.use_l1_loss = not config['no_L1_loss']
            self.use_gan_feat_loss = config['use_gan_feat_loss']
            self.lambda_gan_feat = config['lambda_gan_feat']
            self.use_lpips_loss = config['use_lpips_loss']
            self.lambda_lpips = config['lambda_lpips']
            self.use_cycle_loss = config['use_cycle_loss']
            self.lambda_cycle_forward = config['lambda_cycle_forward']
            self.lambda_cycle_backward = config['lambda_cycle_backward']

            # define D
            self.netD = define_discriminator_from_config(config)
            self.netD.to(self.device)
            self.discriminator_networks.append(self.netD)

            # define F
            if self.use_cycle_loss:
                self.output_nc = config['output_nc']
                self.netF = define_generator(self.output_nc, g_input_nc, config['ngf'], config['netG'],
                                             config['norm_type'], not config['no_dropout'], config['init_type'],
                                             config['init_gain'])
                self.netF.to(self.device)
                self.generator_networks.append(self.netF)
                self.model_names.extend(['F'])
                self.trace_model_names.extend(['F'])

            # losses
            if not self.use_l1_loss:
                self.loss_names.remove('G_L1')

            if self.use_gan_feat_loss:
                self.n_layers_D = config['n_layers_D']

                self.criterion_gan_feat = torch.nn.L1Loss()
                self.loss_names.extend(['G_GAN_Feat'])

            if self.use_lpips_loss:
                self.criterion_lpips = lpips.LPIPS(net='vgg').to(self.device)
                self.loss_names.extend(['G_lpips'])

            if self.use_cycle_loss:
                self.criterionCycle = torch.nn.L1Loss()
                self.netD_F = define_discriminator(g_input_nc + self.output_nc,
                                                   config['ndf'],
                                                   config['netD'],
                                                   config['norm_type'],
                                                   config['init_type'],
                                                   config['init_gain'],
                                                   config['num_D'] if 'num_D' in config else None,
                                                   config['n_layers_D'] if 'n_layers_D' in config else None,
                                                   config['accessible_intermediate_feat'])
                self.netD_F.to(self.device)
                self.discriminator_networks.append(self.netD_F)
                self.loss_names.extend(['F_GAN', 'cycle_forward', 'cycle_backward', 'D_F_real', 'D_F_fake'])
                self.visual_names.extend(['fake_feature', 'rec_feature', 'rec_color', 'rec_depth'])
                self.model_names.extend(['D_F'])

                # add and modify visuals
                if self.use_mapping_network:
                    self.visual_names.append('fake_mapped_feature_example')
                    self.visual_names.append('rec_mapped_feature_example')
                if self.use_mapping_network and not self.use_mapping_feat_concatenation:
                    self.visual_names.remove('fake_feature')
                    self.visual_names.remove('rec_feature')

        def visual_sort(name):
            splitted = name.split('_')
            name = ''.join(splitted[1:]) + splitted[0]
            return name
        self.visual_names.sort(key=visual_sort)

        # define optimizers
        if self.initial_model_mode is ModelMode.Train:
            generator_parameters = [g.parameters() for g in self.generator_networks]
            discriminator_parameters = [d.parameters() for d in self.discriminator_networks]
            self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(generator_parameters), lr=config['lr'],
                                                betas=(config['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain.from_iterable(discriminator_parameters),
                                                lr=config['lr'], betas=(config['beta1'], 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.schedulers = [get_scheduler(config, optimizer, 0) for optimizer in self.optimizers]

        # declare necessary members
        self.real_m_input = None
        self.real_g_input = None
        self.rec_color = None
        self.rec_depth = None
        self.rec_color_depth = None
        self.fake_feature = None
        self.rec_feature = None
        self.real_mapped_feature_example = None
        self.fake_mapped_feature_example = None
        self.rec_mapped_feature_example = None
        self.loss_D_F_real = None
        self.loss_D_F_fake = None
        self.loss_D_F = None

    def print_summary(self, image_size: tuple):
        g_input_nc = self.input_nc
        if self.use_mapping_network:
            if self.use_mapping_feat_concatenation:
                g_input_nc = self.nmf + 1
            else:
                g_input_nc = self.nmf
        print("Summarize_net_G")
        summary(self.netG, (g_input_nc, image_size[0], image_size[1]))

        print("Summarize_net_D")
        summary(self.netD, (self.input_nc + self.output_nc, image_size[0], image_size[1]))

        if self.use_mapping_network:
            print("Summarize_net_M")
            metadata = {'image_size': image_size, 'device': self.device}
            try:
                summary(self.netM, self.netM.input_noise(metadata).shape)
            except:
                print('Cannot create summary of mapping network of type {}'.format(self.netM_type))

        if self.use_cycle_loss:
            print("Summarize_net_F")
            summary(self.netM, (self.output_nc, image_size[0], image_size[1]))

            print("Summarize_net_D_F")
            summary(self.netD_F, (g_input_nc + self.output_nc, image_size[0], image_size[1]))

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        Pix2PixModel.compute_visuals(self)

        if self.use_mapping_network:
            self.real_mapped_feature_example = torch.clone(self.real_g_input[:, 0:1]).detach()

        if self.initialize_complete_model and self.use_cycle_loss:
            # compute color and depth visuals
            self.rec_color = torch.index_select(self.rec_color_depth, 1, torch.tensor([0, 1, 2]).to(self.device))
            self.rec_depth = torch.index_select(self.rec_color_depth, 1, torch.tensor([3]).to(self.device))

            if self.use_mapping_network:
                self.fake_mapped_feature_example = torch.clone(self.fake_feature[:, 0:1]).detach()
                self.rec_mapped_feature_example = torch.clone(self.rec_feature[:, 0:1]).detach()

            if self.use_mapping_network and self.use_mapping_feat_concatenation:
                self.fake_feature = torch.clone(self.fake_feature[:, self.nmf - 1: self.nmf]).detach()
                self.rec_feature = torch.clone(self.rec_feature[:, self.nmf - 1: self.nmf]).detach()

    def forward(self):
        # get G input
        if self.use_mapping_network:
            self.real_m_input = get_mapping_network_input(self.netM_type, self.real_landmarks, self.real_feature)
            self.real_g_input = self.netM(self.real_m_input)  # M(L) = A
            if self.use_mapping_feat_concatenation:
                self.real_g_input = torch.cat((self.real_g_input, self.real_feature), 1)  # concatenate feature map
        else:
            self.real_g_input = self.real_feature  # = A

        # generator forward
        self.fake_color_depth = self.netG(self.real_g_input)  # G(A)

        # apply depth mask if necessary
        if self.use_depth_mask:
            self.fake_color_depth = self.apply_depth_as_mask(self.fake_color_depth)

        if self.initialize_complete_model and self.use_cycle_loss and self.current_mode is not ModelMode.Inference:
            self.forward_end_time = time.time()
            self.fake_feature = self.netF(self.real_color_depth)  # F(B)
            self.rec_feature = self.netF(self.fake_color_depth)  # F(G(A))
            self.rec_color_depth = self.netG(self.fake_feature)  # G(F(B))

            # apply depth mask if necessary
            if self.use_depth_mask:
                self.rec_color_depth = self.apply_depth_as_mask(self.rec_color_depth)

    def backward_D(self):
        # we use conditional GANs; we need to feed both input and output to the discriminator
        loss = 0

        # D
        input_D_real = torch.cat((self.real_feature, self.real_color_depth), 1)
        input_D_fake = torch.cat((self.real_feature, self.fake_color_depth), 1)
        self.loss_D, self.loss_D_real, self.loss_D_fake = self.discriminator_losses(self.netD, input_D_real,
                                                                                    input_D_fake)
        loss += self.loss_D

        if self.use_cycle_loss:
            # D_F
            input_D_F_real = torch.cat((self.real_g_input, self.real_color_depth), 1)
            input_D_F_fake = torch.cat((self.fake_feature, self.real_color_depth), 1)
            self.loss_D_F, self.loss_D_F_real, self.loss_D_F_fake = self.discriminator_losses(self.netD_F,
                                                                                              input_D_F_real,
                                                                                              input_D_F_fake)
            loss += self.loss_D_F

        loss.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # 0. Get and format necessary data
        input_D = torch.cat((self.real_feature, self.fake_color_depth), 1)
        self.pred_D_fake = self.netD(input_D)

        if isinstance(self.pred_D_fake, list) and not isinstance(self.pred_D_fake[0], list):
            self.pred_D_fake = [self.pred_D_fake]

        self.loss_G = 0

        # 1. G(A) should fake the discriminator
        self.loss_G_GAN = self.criterionGAN(self.pred_D_fake, True)
        self.loss_G += self.loss_G_GAN

        if self.use_l1_loss:
            # 2. G(A) = B L1
            self.loss_G_L1 = self.criterionL1(self.fake_color_depth, self.real_color_depth) * self.lambda_L1
            self.loss_G += self.loss_G_L1

        # 3. gan feature
        if self.use_gan_feat_loss:
            self.loss_G_GAN_Feat = self.calculate_gan_feat_loss()  # applied lambda in calculate_gan_feat_loss()
            self.loss_G += self.loss_G_GAN_Feat

        # 4. Perceptual similarity loss LPIPS
        if self.use_lpips_loss:
            self.loss_G_lpips = self.criterion_lpips.forward(self.fake_color_depth[:, :3],
                                                             self.real_color_depth[:, :3]).item()
            self.loss_G_lpips += self.criterion_lpips.forward(self.fake_color_depth[:, 3],
                                                              self.real_color_depth[:, 3]).item()
            self.loss_G_lpips *= self.lambda_lpips
            self.loss_G += self.loss_G_lpips

        # 5. Cycle Consistency loss
        if self.use_cycle_loss:
            # F GAN loss
            input_D_F = torch.cat((self.fake_feature, self.real_color_depth), 1)
            self.loss_F_GAN = self.criterionGAN(self.netD_F(input_D_F), True)
            # Forward cycle loss || F(G(A)) - A||
            self.loss_cycle_forward = self.criterionCycle(self.rec_feature,
                                                          self.real_g_input) * self.lambda_cycle_forward
            # Backward cycle loss || G(F(B)) - B||
            self.loss_cycle_backward = self.criterionCycle(self.rec_color_depth,
                                                           self.real_color_depth) * self.lambda_cycle_backward

            self.loss_G += self.loss_F_GAN + self.loss_cycle_forward + self.loss_cycle_backward

        self.loss_G.backward()

    def calculate_gan_feat_loss(self):
        # Get and format necessary infos
        input_d_real = torch.cat((self.real_feature, self.real_color_depth), 1)
        with torch.no_grad():
            pred_D_real = self.netD(input_d_real)

        if not isinstance(pred_D_real[0], list):
            pred_D_real = [pred_D_real]

        num_D = len(pred_D_real)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / num_D

        for i in range(num_D):
            for j in range(len(self.pred_D_fake[i]) - 1):
                loss_G_GAN_Feat += (
                        D_weights
                        * feat_weights
                        * self.criterion_gan_feat(self.pred_D_fake[i][j], pred_D_real[i][j].detach())
                        * self.lambda_gan_feat
                )

        return loss_G_GAN_Feat
