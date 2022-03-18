import numpy as np
import torch
import cv2 as cv
from metrics.image_pair_metric import ImagePairMetric
from metrics.simple_statistics_metric import SimpleStatisticsMetric
from util.image_tools import map_image_values


class SSIMMetric(ImagePairMetric, SimpleStatisticsMetric):
    @staticmethod
    def ssim(prediction, target, current_value_range):
        """
        Computes the Structural Similarity between prediction and target.
        Assuming image arrays size is [1, channels, width, height]

        Parameters:
            prediction (torch.Tensor): The predictions.
            target (torch.Tensor): The targets.
            current_value_range (tuple): The current value range of the image data. e.g.(0, 255)
        """
        prediction = prediction[0].cpu().clone().detach().numpy()
        prediction = np.transpose(prediction, (1, 2, 0))

        target = target[0].cpu().clone().detach().numpy()
        target = np.transpose(target, (1, 2, 0))

        prediction = map_image_values(prediction, current_value_range, (0, 255))
        target = map_image_values(target, current_value_range, (0, 255))

        if prediction.shape != target.shape:
            raise ValueError("Input images must have the same dimensions.")

        return SSIMMetric._ssim(prediction, target)

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()

    def add_image_pairs_to_statistics(self, images_real, images_fake):
        if images_real.size() != images_fake.size():
            raise RuntimeError("image arrays do not have same amount of images!")

        images_real = images_real.clone().detach().cpu()
        images_fake = images_fake.clone().detach().cpu()

        for i in range(images_real.size(0)):
            current_real_image = torch.index_select(images_real, 0, torch.tensor([i]))
            current_fake_image = torch.index_select(images_fake, 0, torch.tensor([i]))

            curr_ssim = SSIMMetric.ssim(current_fake_image, current_real_image, (-1, 1))

            self.add_statistics_entry(curr_ssim)
