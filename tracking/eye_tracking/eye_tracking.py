# -*- coding: utf-8 -*-
"""
This Script gives access to the basic Eye tracking functionalities.
"""
import importlib
import sys
from abc import ABC, abstractmethod
from enum import Enum

from config import ConfigOptionPackage, ConfigOptionMetadata, BaseConfig
from .. import LandmarkTracker, LandmarkTrackerInput


class EyeTrackingMethod(Enum):
    """
    All implemented Methods for Eye tracking.
    """
    Infrared = 'Infrared'
    Gradient_Based = 'Gradient_Based'

    def __str__(self):
        return self.value


class EyeTrackingMethodCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(EyeTrackingMethod, 'eye_tracking_method', EyeTrackingMethod.Gradient_Based, 'The eye tracking method to extract the eye landmarks', choices=list(EyeTrackingMethod)),
        ]


class EyeTrackerInput(LandmarkTrackerInput):
    def __init__(self, color_image=None, infrared_image=None, bbox_eye_left=None, bbox_eye_right=None):
        super().__init__(color_image)
        self.infrared_image = infrared_image
        self.bbox_eye_left = bbox_eye_left
        self.bbox_eye_right = bbox_eye_right


class EyeTracker(LandmarkTracker, ABC):
    @abstractmethod
    def track_landmarks(self, input_data: EyeTrackerInput):
        super(EyeTracker, self).track_landmarks(input_data)

    @staticmethod
    def number_of_landmarks():
        return 2


def eye_tracker_class(method: EyeTrackingMethod) -> EyeTracker.__class__:
    name = str(method)
    module_filename = '.' + name.lower() + '_eye_tracker'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    method_class = None

    method_class_name = name.replace('_', '') + 'EyeTracker'
    for name, cls in module.__dict__.items():
        if name.lower() == method_class_name.lower() and issubclass(cls, EyeTracker):
            method_class = cls

    if method_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of EyeTracker with class name that matches %s in lowercase." % (
                module_filename, method_class_name))
    return method_class


def new_eye_tracker_instance(eye_tracking_method: EyeTrackingMethod, config: BaseConfig) -> EyeTracker:
    method_class = eye_tracker_class(eye_tracking_method)
    return method_class(config)

