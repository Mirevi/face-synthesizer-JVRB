import os
import time
from tqdm import tqdm

import models
from config import new_argparse_config, ConfigType, BaseConfig
from config.evaluate_config import EvaluateCOP
from config.train_config import TrainStandardConfig
from data.datasets import dataset_class, new_dataset_data_loader_instance, DatasetMode
from models import ModelMode
from util.hard_drive_util import get_latest_checkpoint_epoch
from util.image_tools import tensor_to_image, map_image_values, save_image


class Evaluate:
    def __init__(self, config: BaseConfig):
        # validate config
        if EvaluateCOP not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # get necessary values from config
        self.input_root = config['input_root']
        self.opt_file_name = config['opt_file_name']
        self.uniform_eval_dataset = config['uniform_eval_dataset']
        self.uniform_depth_mask_usage = config['uniform_depth_mask_usage']

        if self.uniform_eval_dataset:
            self.dataset_options = config.get_options_from_provider(dataset_class(config['dataset_type']))
        if self.uniform_depth_mask_usage:
            self.no_depth_mask = config['no_depth_mask']

        self.evaluation = {}

    # TODO refactor into multiple methods
    def __call__(self):
        start_time = time.time()

        # create evaluation file
        evaluation_file = os.path.join(self.input_root, "evaluations.txt")
        evaluation_writer = open(evaluation_file, 'w')

        # get list of all training runs
        model_name_list = [i for i in os.listdir(self.input_root) if os.path.isdir(os.path.join(self.input_root, i))
                           and os.path.exists(os.path.join(self.input_root, i, self.opt_file_name))]

        for i, model_name in enumerate(model_name_list, 1):
            model_root = os.path.join(self.input_root, model_name)
            options_file_path = os.path.join(model_root, self.opt_file_name)
            image_dir = os.path.join(model_root, 'web/images')

            # read config file
            config_file = open(options_file_path, 'r')
            config_string = config_file.read()
            config_file.close()

            # modify config_string
            if self.uniform_eval_dataset:
                for key, value in self.dataset_options.items():
                    config_string += '\n{}: {}'.format(key, value)
            if self.uniform_depth_mask_usage:
                config_string += '\n{}: {}'.format('no_depth_mask', self.no_depth_mask)

            # get train config from string
            train_config = TrainStandardConfig.from_string(config_string)
            train_config.gather_options()
            train_config.print()

            # no eval dataset given so skip this model
            if not self.uniform_eval_dataset and train_config['no_eval']:
                continue

            # get necessary metadata
            latest_epoch = get_latest_checkpoint_epoch(model_root)

            # load eval dataset
            eval_dataset = new_dataset_data_loader_instance(train_config, DatasetMode.Eval)
            print('The number of evaluation images = {}'.format(len(eval_dataset.dataset)))

            # load model
            model = models.new_model_instance(train_config, ModelMode.Eval)
            model.print_networks()
            model.load_networks(model_root, latest_epoch)

            # evaluate model
            time_delta = time.time() - start_time
            print("Evaluate Model {} [{}/{} Total Time: {:.0f} min {:.0f} sec]".format(model_name, i,
                                                                                       len(model_name_list),
                                                                                       time_delta / 60,
                                                                                       time_delta % 60))
            eval_start_time = time.time()

            # try to prevent that dataset loading time affects forward time metric
            model.set_input(next(eval_dataset.__iter__()))
            model.test()

            # evaluation
            for data in tqdm(eval_dataset):
                model.set_input(data)
                model.evaluate()

            evaluation = model.get_current_evaluation_results()
            print('End of Evaluation: Time Taken: {:.0f} sec'.format(time.time() - eval_start_time))

            # save evaluation data
            self.evaluation[model_name] = evaluation

            # save to evaluation file
            evaluation_string = self.evaluation_to_string(self.evaluation[model_name])
            evaluation_string = evaluation_string.replace('\n', '\n\t')
            evaluation_writer.write('Model Name: {}\n\t{}\n'.format(model_name, evaluation_string))
            evaluation_writer.flush()

        # close evaluation writer
        evaluation_writer.close()

        time_delta = time.time() - start_time
        print('Total Time Taken: {:.0f} min {:.0f} sec for {} models'.format(
            time_delta / 60, time_delta % 60, len(model_name_list)))

    @staticmethod
    def evaluation_to_string(evaluation):
        message = ''
        for metric_name, metric_dict in evaluation.items():
            message += metric_name + '\n'
            for value_name, value in metric_dict.items():
                message += '\t{}: {:.4f}\n'.format(value_name, float(value))
        return message


if __name__ == "__main__":
    script_config = new_argparse_config(ConfigType.Evaluate)
    script_config.gather_options()
    script_config.print()
    script_config_save_path = os.path.join(script_config['input_root'])
    script_config.save_to_disk(script_config_save_path, 'EvalConfig.txt')

    evaluate_script = Evaluate(script_config)
    evaluate_script()
