import sys
from enum import Enum
import importlib
from .base_metric import BaseMetric


class MetricType(Enum):
    Forward_Time = 'Forward_Time'
    Pixel_Accuracy = 'Pixel_Accuracy'
    Threshold_Pixel_Accuracy = 'Threshold_Pixel_Accuracy'
    PSNR = 'PSNR'
    SSIM = 'SSIM'
    FID = 'FID'
    LPIPS = 'LPIPS'

    def __str__(self):
        return self.value


def metric_class(metric_type: MetricType):
    name = str(metric_type)
    module_filename = '.' + name.lower() + '_metric'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    type_class = None

    type_class_name = name.replace('_', '') + 'metric'
    for name, cls in module.__dict__.items():
        if name.lower() == type_class_name.lower() and issubclass(cls, BaseMetric):
            type_class = cls

    if type_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseMetric with class name that matches %s in lowercase." % (
                module_filename, type_class_name))
    return type_class


def new_metric_instance(metric_type: MetricType, config):
    type_class = metric_class(metric_type)
    return type_class(config)

