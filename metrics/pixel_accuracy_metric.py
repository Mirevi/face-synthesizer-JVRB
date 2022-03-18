import torch
from metrics.image_pair_metric import ImagePairMetric
from metrics.simple_statistics_metric import SimpleStatisticsMetric


class PixelAccuracyMetric(ImagePairMetric, SimpleStatisticsMetric):
    @staticmethod
    def pixel_accuracy(prediction, target):
        """
        Computes the pixel_accuracy between the prediction and the target.

        Parameters:
            prediction (torch.Tensor): The predictions.
            target (torch.Tensor): The targets.
        """
        return (prediction == target).sum().item() / target.numel()

    def add_image_pairs_to_statistics(self, images_real, images_fake):
        if images_real.size() != images_fake.size():
            raise RuntimeError("image arrays do not have same amount of images!")

        images_real = images_real.clone().detach().cpu()
        images_fake = images_fake.clone().detach().cpu()

        for i in range(images_real.size(0)):
            current_real_image = torch.index_select(images_real, 0, torch.tensor([i]))
            current_fake_image = torch.index_select(images_fake, 0, torch.tensor([i]))
            curr_pixel_acc = PixelAccuracyMetric.pixel_accuracy(current_fake_image, current_real_image)

            self.add_statistics_entry(curr_pixel_acc)
