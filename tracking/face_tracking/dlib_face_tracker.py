# -*- coding: utf-8 -*-
"""
A dlib Face Tracker.
"""
import os.path
import sys

import dlib
import numpy as np

from tracking.face_tracking import FaceTrackerInput
from tracking.face_tracking.face_tracking import FaceTracker


class DLibFaceTracker(FaceTracker):
    @staticmethod
    def number_of_landmarks():
        return 68

    def __init__(self, config):
        super().__init__(config)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            os.path.join('.', sys.modules[__name__].__package__.replace('.', '\\'),
                         "shape_predictor_68_face_landmarks.dat"))

    def track_landmarks(self, input_data: FaceTrackerInput):
        super(DLibFaceTracker, self).track_landmarks(input_data)
        image = self.input.image

        self.tracked_data = np.zeros((68, 2))

        faces = self.detector(image)

        face = faces[0]
        landmarks = self.predictor(image, face)

        for n in range(0, 68):
            self.tracked_data[n, 0] = landmarks.part(n).x
            self.tracked_data[n, 1] = landmarks.part(n).y

        return self.tracked_data
