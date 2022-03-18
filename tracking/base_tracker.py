from abc import ABC

from config import ConfigPackageProvider


class BaseTracker(ConfigPackageProvider, ABC):
    """
    A Tracker which tracks some value from a given input.
    """

    def __init__(self, config):
        # validate config
        if self not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        self.input = None
        self.tracked_data = None
