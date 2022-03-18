import torch
from metrics.image_pair_metric import ImagePairMetric
from metrics.simple_statistics_metric import SimpleStatisticsMetric
from util.image_tools import map_image_values


class PSNRMetric(ImagePairMetric, SimpleStatisticsMetric):
    @staticmethod
    def psnr(prediction, target, current_value_range):
        """
        Computes the Peak Signal-to-Noise Ratio between prediction and target.

        Parameters:
            prediction (torch.Tensor): The predictions.
            target (torch.Tensor): The targets.
            current_value_range (tuple): The current value range of the image data. e.g.(0, 255)
        """
        prediction = map_image_values(prediction.clone(), current_value_range, (0, 255))
        target = map_image_values(target.clone(), current_value_range, (0, 255))

        mse = torch.mean((target - prediction) ** 2)
        if mse == 0:
            return float('inf')
        else:
            return 20 * torch.log10(255.0 / torch.sqrt(mse))

    def add_image_pairs_to_statistics(self, images_real, images_fake):
        if images_real.size() != images_fake.size():
            raise RuntimeError("image arrays do not have same amount of images!")

        images_real = images_real.clone().detach().cpu()
        images_fake = images_fake.clone().detach().cpu()

        for i in range(images_real.size(0)):
            current_real_image = torch.index_select(images_real, 0, torch.tensor([i]))
            current_fake_image = torch.index_select(images_fake, 0, torch.tensor([i]))

            curr_psnr = PSNRMetric.psnr(current_fake_image, current_real_image, (-1, 1))

            self.add_statistics_entry(curr_psnr)
