# -*- coding: utf-8 -*-
"""
An Eye Tracker can get landmarks of the eyes from an image tensor.
"""
import cv2 as cv
import numpy as np

from config import ConfigOptionMetadata, ConfigOptionPackage
from tracking.eye_tracking import EyeTrackerInput
from tracking.eye_tracking.eye_tracking import EyeTracker


class InfraredEyeTrackerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'eye_tracking_threshold', 88,
                                 'The eye tracking threshold if infrared tracking is used.'),
        ]


class InfraredEyeTracker(EyeTracker):
    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(InfraredEyeTracker, InfraredEyeTracker).get_required_option_packages()
        packages.extend([InfraredEyeTrackerCOP])
        return packages

    def __init__(self, config):
        super().__init__(config)
        self.eye_tracking_threshold = config['eye_tracking_threshold']

    def track_landmarks(self, input_data: EyeTrackerInput):
        super(InfraredEyeTracker, self).track_landmarks(input_data)
        self.input = input_data
        image = np.copy(self.input.image)
        bbox_eye_left = self.input.bbox_eye_left
        bbox_eye_right = self.input.bbox_eye_right

        x = int(bbox_eye_left["x"])
        y = int(bbox_eye_left["y"])
        w = int((bbox_eye_right["x"] + bbox_eye_right["width"]) - bbox_eye_left["x"])
        h = int(
            bbox_eye_left["height"] if bbox_eye_left["height"] > bbox_eye_right["height"] else bbox_eye_right["height"])

        image = cv.equalizeHist(image)
        roi = image[y:y + h, x:x + w]
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        roi[:, int(w / 3):int(w / 3) * 2] = 255

        # print_numpy(roi, True, True)
        roi = cv.GaussianBlur(roi, (11, 11), 0)
        thresh = self.eye_tracking_threshold
        _, roi = cv.threshold(roi, thresh, 255, cv.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        roi = cv.dilate(roi, kernel, iterations=2)

        roi_left = roi[:, 0:int(w / 2)]
        roi_right = roi[:, int(w / 2):w]

        roi = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)

        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0

        contours, _ = cv.findContours(roi_left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        for cnt in contours:
            (x1, y1, w1, h1) = cv.boundingRect(cnt)
            cv.rectangle(roi, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            y1 += y + int(h1 / 2)
            x1 += x + w1 - 15  # *2

            image[y1 - 3:y1 + 3, x1 - 3:x1 + 3] = np.array([0, 255, 0])

            break

        contours, _ = cv.findContours(roi_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        offset_w = int(w / 2)
        for cnt in contours:
            (x2, y2, w2, h2) = cv.boundingRect(cnt)
            cv.rectangle(roi, (x2 + offset_w, y2), (x2 + w2 + offset_w, y2 + h2), (0, 255, 0), 2)

            y2 += y + int(h2 / 2)
            x2 += x + int(w / 2) + 15

            image[y2 - 3:y2 + 3, x2 - 3:x2 + 3] = np.array([0, 255, 0])

            break

        if x1 == 0 and y1 == 0:
            y1 += y + int(h / 2)
            x1 += x + int(w / 4)

        if x2 == 0 and y2 == 0:
            y2 += y + int(h / 2)
            x2 += x + int(w / 4) * 3

        self.tracked_data = np.asarray([[x1, y1], [x2, y2]])
        return self.tracked_data
