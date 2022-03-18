import numpy as np

from config import ConfigOptionPackage, ConfigOptionMetadata
from tracking.eye_tracking import EyeTracker, EyeTrackerInput
from tracking.eye_tracking.PupilDetector import GradientIntersect
from util.image_tools import convert_into_grayscale, get_image_excerpt


class GradientBasedEyeTrackerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'eye_tracking_stride', 1,
                                 'The stride value determines the accuracy and the speed of the eye tracking. '
                                 'Higher value cause the eye tracking to be faster but less accurate by consider '
                                 'less pixels during calculation.'),
        ]


class GradientBasedEyeTracker(EyeTracker):
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(GradientBasedEyeTracker, GradientBasedEyeTracker).get_required_option_packages()
        packages.extend([GradientBasedEyeTrackerCOP])
        return packages

    def __init__(self, config):
        super().__init__(config)
        self.gradient_intersect = GradientIntersect()
        self.stride = config['eye_tracking_stride']

    def track_landmarks(self, input_data: EyeTrackerInput):
        super(GradientBasedEyeTracker, self).track_landmarks(input_data)
        self.input = input_data
        image = self.input.image
        bbox_eye_left = self.input.bbox_eye_left
        bbox_eye_right = self.input.bbox_eye_right

        image_eye_left = get_image_excerpt(image, bbox_eye_left)
        image_eye_right = get_image_excerpt(image, bbox_eye_right)

        image_eye_left = convert_into_grayscale(image_eye_left)
        image_eye_right = convert_into_grayscale(image_eye_right)

        location_eye_left = self.gradient_intersect.locate(image_eye_left, accuracy=self.stride)
        location_eye_right = self.gradient_intersect.locate(image_eye_right, accuracy=self.stride)

        self.tracked_data = np.asarray([[location_eye_left[1] + bbox_eye_left['x'],
                                         location_eye_left[0] + bbox_eye_left['y']],
                                        [location_eye_right[1] + bbox_eye_right['x'],
                                         location_eye_right[0] + bbox_eye_right['y']]])
        return self.tracked_data
