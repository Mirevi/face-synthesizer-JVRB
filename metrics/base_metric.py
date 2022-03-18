from abc import ABC, abstractmethod

from config import ConfigPackageProvider


class BaseMetric(ConfigPackageProvider, ABC):
    def __init__(self, config):
        # validate config
        if self not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

    @abstractmethod
    def clear_statistics(self):
        """
        Clears the current statistic data.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Restores the initial state of the metric.
        """
        pass

    @abstractmethod
    def __call__(self):
        """
        Computes the metric value with all values from the current statistics.
        """
        pass
