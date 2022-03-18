import os
from argparse import ArgumentParser

import numpy as np
import time

import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from config import ConfigPackageProvider, ConfigOptionPackage, ConfigOptionMetadata
from config.config_option_packages import DepthMaskCOP
from . import networks
from enum import Enum
from metrics import MetricType, new_metric_instance, metric_class
from .networks import TraceableNetwork, SchedulerCOP


class ModelMode(Enum):
    Train = 'Train'
    Eval = 'Eval'
    Inference = 'Inference'

    def __str__(self):
        return self.value


class BaseModelCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'with_train_options', False, 'train options are added to config if true.',
                                 is_constant=True),
        ]


class AdamOptimizerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(float, 'beta1', 0.5, 'momentum term of adam'),
            ConfigOptionMetadata(float, 'lr', 0.0002, 'initial learning rate for adam'),
        ]


class BaseModel(ConfigPackageProvider, ABC):
    """This class is an abstract base class (ABC) for models."""
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(BaseModel, BaseModel).get_required_option_packages()
        packages.extend([BaseModelCOP, DepthMaskCOP, SchedulerCOP])
        return packages

    @staticmethod
    def get_required_providers() -> list:
        return [metric_class(MetricType.Forward_Time)]

    def __init__(self, config, initial_model_mode: ModelMode):
        # validate config
        if self not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # set values from config
        self.config = config
        self.initial_model_mode = initial_model_mode
        self.current_mode = initial_model_mode
        self.beta1 = config['beta1']
        self.lr = config['lr']
        self.lr_policy = config['lr_policy']

        # set other necessary values
        torch.backends.cudnn.benchmark = True

        # initialize members
        self.initialize_complete_model = False if self.current_mode is ModelMode.Inference else True
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # specify the training losses you want to return when BaseModel.get_current_losses is called.
        self.loss_names = []
        # specify the models you want to save to the disk when BaseModel.save_networks is called.
        self.model_names = []
        # specify the models you want to save to the disk when BaseModel.save_traced_networks is called.
        self.trace_model_names = []
        # specify the images you want to return when BaseModel.get_current_visuals is called.
        self.visual_names = []
        # specify the metrics you want to return when BaseModel.get_current_evaluation is called.
        self.metric_names = ['forward_time']
        # optimizers of this model
        self.optimizers = []
        self.lr_plateau_metric = 0  # used for learning rate policy 'plateau'
        self.forward_start_time = 0
        self.forward_end_time = 0
        self.depth_mask_thresh = 1 / 127.5 - 1

        if self.initialize_complete_model:
            self.metric_forward_time = new_metric_instance(MetricType.Forward_Time, config)

    @abstractmethod
    def print_summary(self, image_size: tuple):
        pass

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        if self.current_mode is not ModelMode.Train:
            self.set_mode(ModelMode.Train)

    def evaluate(self):
        """Evaluates the model by metrics; called at the end of every epoch"""
        if self.current_mode is not ModelMode.Eval:
            self.set_mode(ModelMode.Eval)

        with torch.no_grad():
            self.forward_start_time = time.time()
            self.forward()
            if self.forward_end_time < self.forward_start_time:  # declaration during forward is possible
                self.forward_end_time = time.time()

            self.compute_visuals()

            # add time needed to forward time metric
            time_needed = [(self.forward_end_time - self.forward_start_time) * 1000]  # in ms
            self.metric_forward_time.add_forward_times(np.array(time_needed))

    def set_mode(self, mode: ModelMode):
        if not self.initialize_complete_model and (mode is ModelMode.Train or mode is ModelMode.Eval):
            raise RuntimeError('Cannot set the mode to {} because the model is not completely initialized and so the '
                               'necessary resources are not available.'.format(mode))

        """Sets the mode for the model"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if mode is ModelMode.Train:
                    net.train()
                elif mode is ModelMode.Eval or mode is ModelMode.Inference:
                    net.eval()
        self.current_mode = mode

    # TODO rename
    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def apply_depth_as_mask(self, color_depth):
        depth_mask = torch.where(color_depth[:, 3, :, :] >= self.depth_mask_thresh, 1, 0)
        color_depth = (color_depth + 1) * depth_mask - 1
        return color_depth

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.lr_policy == 'plateau':
                scheduler.step(self.lr_plateau_metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_evaluation_results(self):
        """Return evaluation and reset all metrics. train.py will print out these metrics,
        save them to a file and display them with HTML"""
        visual_ret = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                inst = getattr(self, 'metric_' + name)
                visual_ret[name] = inst()
                inst.clear_statistics()

        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, checkpoints_location, epoch: int):
        """Save all the networks to the disk."""
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{}_net_{}.pth'.format(str(epoch), name)
                save_path = os.path.join(checkpoints_location, save_filename)
                net = getattr(self, 'net' + name)
                net.cpu()
                torch.save(net.state_dict(), save_path)
                net.to(self.device)

    def save_traced_networks(self, checkpoints_location, image_size):
        mode = self.current_mode
        self.set_mode(ModelMode.Inference)
        for name in self.trace_model_names:
            if isinstance(name, str):
                save_filename = 'tracedGenerator_net_{}.zip'.format(name)
                save_path = os.path.join(checkpoints_location, save_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, TraceableNetwork):
                    metadata = {"image_size": image_size, "device": self.device}
                    noise = net.input_noise(metadata)
                    traced = torch.jit.trace(net.eval(), noise)
                    traced.save(save_path)
                else:
                    raise RuntimeError("Cant trace Model with name {}. The class of the Model does not inherit from "
                                       "Traceable class".format('net' + name))

        self.set_mode(mode)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, checkpoints_location, epoch: int):
        """Load all the networks from the disk."""
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '{}_net_{}.pth'.format(str(epoch), name)
                load_path = os.path.join(checkpoints_location, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('Loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

        if self.initial_model_mode is ModelMode.Train:
            self.schedulers = [networks.get_scheduler(self.config, optimizer, epoch) for optimizer in self.optimizers]

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
