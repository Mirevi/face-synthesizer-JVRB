import numpy as np
import lpips
import torch
from metrics.image_pair_metric import ImagePairMetric
from metrics.simple_statistics_metric import SimpleStatisticsMetric


class LPIPSMetric(ImagePairMetric, SimpleStatisticsMetric):
    """
    FROM: https://github.com/richzhang/PerceptualSimilarity
    """

    def __init__(self, config, initial_buffer_size=1000):
        SimpleStatisticsMetric.__init__(self, config, initial_buffer_size)

        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
        self.lpips = lpips.LPIPS(net='alex').to(self.device)

    def add_image_pairs_to_statistics(self, images_real, images_fake):
        if images_real.size() != images_fake.size():
            raise RuntimeError("image arrays do not have same amount of images!")

        images_real = images_real.clone().detach().to(self.device)
        images_fake = images_fake.clone().detach().to(self.device)

        curr_lpips = self.lpips.forward(images_real, images_fake)
        curr_lpips = np.reshape(curr_lpips.detach().cpu().numpy(), curr_lpips.size(0))

        self.add_statistics_entries(curr_lpips)
