from argparse import ArgumentParser

import torch

from config import ConfigOptionPackage, ConfigOptionMetadata
from metrics import BaseMetric
from metrics.image_pair_metric import ImagePairMetric
from metrics.simple_statistics_metric import SimpleStatisticsMetric


class ThresholdPixelAccuracyMetricCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'pixel_acc_thresh', 5, 'The Threshold for the Thresholded Pixel Accuracy. Every Channel value of a Pixel is checked by: Math.abs(prediction, target) <= threshold. Threshold range is [0, 255].', choices=range(0, 256)),
        ]


class ThresholdPixelAccuracyMetric(ImagePairMetric, SimpleStatisticsMetric):
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(ThresholdPixelAccuracyMetric, ThresholdPixelAccuracyMetric).get_required_option_packages()
        packages.append(ThresholdPixelAccuracyMetricCOP)
        return packages

    def __init__(self, config, initial_buffer_size=1000):
        SimpleStatisticsMetric.__init__(self, config, initial_buffer_size)

        self.threshold = config['pixel_acc_thresh'] / 127.5

    def add_image_pairs_to_statistics(self, images_real, images_fake):
        if images_real.size() != images_fake.size():
            raise RuntimeError("image arrays do not have same amount of images!")

        images_real = images_real.clone().detach().cpu()
        images_fake = images_fake.clone().detach().cpu()

        for i in range(images_real.size(0)):
            current_real_image = torch.index_select(images_real, 0, torch.tensor([i]))
            current_fake_image = torch.index_select(images_fake, 0, torch.tensor([i]))

            curr_pixel_acc = self.threshold_pixel_accuracy(current_fake_image, current_real_image)

            self.add_statistics_entry(curr_pixel_acc)

    def threshold_pixel_accuracy(self, prediction, target):
        """
        Computes the threshold pixel_accuracy between the prediction and the target.
        The mathematical criteria is: abs(prediction - target) <= threshold

        Parameters:
            prediction (torch.Tensor): The predictions.
            target (torch.Tensor): The targets.
        """
        return ((prediction - target).abs() <= self.threshold).sum().item() / target.numel()
