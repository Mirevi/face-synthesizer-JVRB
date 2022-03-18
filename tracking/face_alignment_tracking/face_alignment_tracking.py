import importlib
import sys
from abc import ABC, abstractmethod
from enum import Enum

from config import ConfigOptionPackage, ConfigOptionMetadata, BaseConfig
from tracking import BaseTracker


class FaceAlignmentTrackingMethod(Enum):
    SynergyNet = 'SynergyNet'

    def __str__(self):
        return self.value


class FaceAlignmentTrackingMethodCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(FaceAlignmentTrackingMethod, 'face_alignment_tracking_method',
                                 FaceAlignmentTrackingMethod.SynergyNet,
                                 'The face alignment tracking method to extract the face alignment direction',
                                 choices=list(FaceAlignmentTrackingMethod)),
        ]


class FaceAlignmentTrackerInput:
    def __init__(self, image=None, face_bounding_box=None):
        self.image = image
        self.face_bounding_box = face_bounding_box


class FaceAlignmentTracker(BaseTracker, ABC):
    @abstractmethod
    def track_face_alignment(self, input_data: FaceAlignmentTrackerInput):
        self.input = input_data


def face_alignment_tracker_class(method: FaceAlignmentTrackingMethod) -> FaceAlignmentTracker.__class__:
    name = str(method)
    module_filename = '.' + name.lower() + '_face_alignment_tracker'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    method_class = None

    method_class_name = name.replace('_', '') + 'FaceAlignmentTracker'
    for name, cls in module.__dict__.items():
        if name.lower() == method_class_name.lower() and issubclass(cls, FaceAlignmentTracker):
            method_class = cls

    if method_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of FaceAlignmentTracker with class "
            "name that matches %s in lowercase." % (module_filename, method_class_name))
    return method_class


def new_face_alignment_tracker_instance(method: FaceAlignmentTrackingMethod,
                                        config: BaseConfig) -> FaceAlignmentTracker:
    method_class = face_alignment_tracker_class(method)
    return method_class(config)
