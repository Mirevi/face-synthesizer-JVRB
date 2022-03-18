from abc import ABC, abstractmethod

from metrics import BaseMetric


class ImagePairMetric(BaseMetric, ABC):
    @abstractmethod
    def add_image_pairs_to_statistics(self, images_real, images_fake):
        """
        Adds given image pairs to the statistics of this metric.
        The metric assumes that the input data is in range [-1 ; 1].

        Parameters:
            images_real (torch.Tensor) - A torch Tensor with data from the real image distribution.
            images_fake (torch.Tensor) - A torch Tensor with data from the fake image distribution.
        """
        pass
