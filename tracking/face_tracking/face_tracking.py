# -*- coding: utf-8 -*-
"""
This Script gives access to the basic Face tracking functionalities.
"""
import importlib
import sys
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from config import ConfigOptionPackage, ConfigOptionMetadata, BaseConfig
from .. import LandmarkTracker, LandmarkTrackerInput


class FaceTrackingMethod(Enum):
    """
    All implemented Methods for Face tracking.
    """
    FAN = 'FAN'
    DLib = 'DLib'

    def __str__(self):
        return self.value


class FaceTrackingMethodCop(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(FaceTrackingMethod, 'face_tracking_method', FaceTrackingMethod.FAN, 'The face tracking method to extract the face landmarks', choices=list(FaceTrackingMethod)),
        ]


class FaceTrackerInput(LandmarkTrackerInput):
    def __init__(self, image=None):
        super().__init__(image)


class FaceTracker(LandmarkTracker, ABC):
    @abstractmethod
    def track_landmarks(self, input_data: FaceTrackerInput):
        super(FaceTracker, self).track_landmarks(input_data)


def face_tracker_class(method: FaceTrackingMethod) -> FaceTracker.__class__:
    name = str(method)
    module_filename = '.' + name.lower() + '_face_tracker'
    module = importlib.import_module(module_filename, vars(sys.modules[__name__])['__package__'])
    method_class = None

    method_class_name = name.replace('_', '') + 'FaceTracker'
    for name, cls in module.__dict__.items():
        if name.lower() == method_class_name.lower() and issubclass(cls, FaceTracker):
            method_class = cls

    if method_class is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of FaceTracker with class name that matches %s in lowercase." % (
                module_filename, method_class_name))
    return method_class


def new_face_tracker_instance(face_tracking_method: FaceTrackingMethod, config: BaseConfig) -> FaceTracker:
    method_class = face_tracker_class(face_tracking_method)
    return method_class(config)
