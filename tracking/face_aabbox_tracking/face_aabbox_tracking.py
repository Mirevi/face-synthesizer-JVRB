# aabbox = axis aligned bounding box
import importlib
import sys
from abc import ABC, abstractmethod
from enum import Enum

from config import ConfigOptionPackage, ConfigOptionMetadata, BaseConfig
from tracking import BaseTracker


class FaceAABBoxTrackingMethod(Enum):
    SynergyNet = 'SynergyNet'

    def __str__(self):
        return self.value


class FaceAABBoxTrackingMethodCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(FaceAABBoxTrackingMethod, 'face_aabbox_tracking_method',
                                 FaceAABBoxTrackingMethod.SynergyNet,
                                 'The face axis aligned bounding box tracking method to use.',
                                 choices=list(FaceAABBoxTrackingMethod)),
        ]


class FaceAABBoxTrackerInput:
    def __init__(self, image=None, landmarks=None):
        self.image = image
        self.landmarks = landmarks


class FaceAABBoxTracker(BaseTracker, ABC):
    @abstractmethod
    def track_face_aabbox(self, input_data: FaceAABBoxTrackerInput):
        self.input_data = input_data


def face_aabbox_tracker_class(method: FaceAABBoxTrackingMethod) -> FaceAABBoxTracker.__class__:
    name = str(method)
    module_filename = '.' + name.lower() + '_face_aabbox_tracker'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    method_class = None

    method_class_name = name.replace('_', '') + 'FaceAABBoxTracker'
    for name, cls in module.__dict__.items():
        if name.lower() == method_class_name.lower() and issubclass(cls, FaceAABBoxTracker):
            method_class = cls

    if method_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of FaceAABBoxTracker with class "
            "name that matches %s in lowercase." % (module_filename, method_class_name))
    return method_class


def new_face_aabbox_tracker_instance(method: FaceAABBoxTrackingMethod, config: BaseConfig) -> FaceAABBoxTracker:
    method_class = face_aabbox_tracker_class(method)
    return method_class(config)
