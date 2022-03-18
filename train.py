"""General-purpose training script for image-to-image translation."""
import os
import sys

import time
from shutil import rmtree

import torch
from tqdm import tqdm

from config import new_argparse_config, ConfigType
from config.train_config import TrainPackageProvider
from data.datasets import new_dataset_data_loader_instance, DatasetMode
from models import new_model_instance, ModelMode
from documentation import Documentation
from util.hard_drive_util import get_latest_checkpoint_epoch


class Train:
    def __init__(self, config):
        # validate config
        if TrainPackageProvider not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # get necessary values from config
        self.experiment_name = config['name']
        self.output_root = config['output_root']
        self.continue_process = config['continue_process']
        self.continue_epoch = config['continue_epoch']
        self.overwrite = config['overwrite']
        self.n_epochs = config['n_epochs']
        self.n_epochs_decay = config['n_epochs_decay']
        self.save_epoch_freq = config['save_epoch_freq']
        self.image_size = config['dataset_image_size']
        self.with_eval = not config['no_eval']  # TODO + automatic check if eval exists in dataset
        self.eval_epoch_freq = config['eval_epoch_freq']

        # define locations
        self.checkpoint_location = os.path.join(self.output_root, self.experiment_name)
        self.initial_epoch = 1

        # if overwrite then delete current output content
        if self.overwrite:
            rmtree(self.checkpoint_location, ignore_errors=True)

        if not self.continue_process and os.path.exists(self.checkpoint_location):
            raise RuntimeError("A model with name '{}' already exists. If you want to overwrite the model set the "
                               "overwrite option of the config to True.")

        # save config file
        script_config.save_to_disk(self.checkpoint_location, 'TrainConfig.txt')

        # declare or create necessary members
        # datasets
        self.train_dataset = new_dataset_data_loader_instance(config, DatasetMode.Train)
        print('The number of training images = {}'.format(len(self.train_dataset)))
        if self.with_eval:
            self.eval_dataset = new_dataset_data_loader_instance(config, DatasetMode.Eval)
            print('The number of evaluation images = {}'.format(len(self.eval_dataset)))
        self.model = new_model_instance(config, ModelMode.Train)
        self.model.print_networks()
        self.documentation = Documentation(config, self.continue_process)

        # load networks if continue training
        if self.continue_process:
            self.initial_epoch = get_latest_checkpoint_epoch(self.checkpoint_location, self.continue_epoch) + 1
            self.model.load_networks(self.checkpoint_location, self.initial_epoch - 1)

    def __call__(self):
        train_start_time = time.time()
        for epoch in range(self.initial_epoch, self.n_epochs + self.n_epochs_decay + 1):
            epoch_start_time = time.time()
            for epoch_iter, data in enumerate(self.train_dataset):
                self.model.set_input(data)
                self.model.optimize_parameters()
                self.model.compute_visuals()

                epoch_progress = epoch_iter / len(self.train_dataset)
                visuals = self.model.get_current_visuals()
                losses = self.model.get_current_losses()
                self.documentation.document_iteration(epoch, epoch_iter + 1, epoch_progress, visuals, losses)

            self.model.update_learning_rate()

            if epoch % self.save_epoch_freq == 0:
                self.model.save_networks(self.checkpoint_location, epoch)
                self.model.save_traced_networks(self.checkpoint_location, self.image_size)

            print('End of epoch {} / {} \t Time Taken: {:.0f} sec'.format(
                epoch, self.n_epochs + self.n_epochs_decay, time.time() - epoch_start_time))

            if self.with_eval and (epoch % self.eval_epoch_freq == 0 or epoch == self.n_epochs + self.n_epochs_decay):
                print('evaluate the model at the end of epoch {}.'.format(epoch))
                eval_start_time = time.time()

                for data in tqdm(self.eval_dataset):
                    self.model.set_input(data)
                    self.model.evaluate()

                evaluation = self.model.get_current_evaluation_results()

                # document evaluation
                self.documentation.document_evaluation(epoch, evaluation)

                # document visuals
                visuals = self.model.get_current_visuals()
                self.documentation.document_visuals(visuals, epoch)

                print('End of Evaluation: Epoch %d \t Time Taken: %d sec' % (epoch, time.time() - eval_start_time))

        train_end_time = time.time()
        seconds_delta = train_end_time - train_start_time
        days = int(seconds_delta / (60*60*24))
        hours = int(seconds_delta / (60*60)) % 24
        minutes = int(seconds_delta / 60) % 60
        seconds = int(seconds_delta) % 60
        print('End of Training: Time Taken: {} d {} h {} m {} s'.format(days, hours, minutes, seconds))


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(1.0)

    # TODO if continue_process load config from file and edit necessary values

    script_config = new_argparse_config(ConfigType.Train)
    script_config.gather_options()
    script_config.print()

    train_script = Train(script_config)
    train_script()
