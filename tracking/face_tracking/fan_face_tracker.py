# -*- coding: utf-8 -*-
"""
A Face Alignment networks Face Tracker.
"""
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType

from tracking.face_tracking import FaceTrackerInput
from tracking.face_tracking.face_tracking import FaceTracker


class FANFaceTracker(FaceTracker):
    @staticmethod
    def number_of_landmarks():
        return 68

    def __init__(self, config):
        super().__init__(config)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

    def track_landmarks(self, input_data: FaceTrackerInput):
        super(FANFaceTracker, self).track_landmarks(input_data)
        image = self.input.image

        with torch.no_grad():  # Important! Otherwise destroys the autograd process
            self.tracked_data = self.network.get_landmarks(image[:, :, 0:3])

        '''
        faces = len(self.tracked_data
        self.tracked_data = np.array(self.tracked_data).reshape((68 * faces, 2))
        self.show_latest_2d_landmarks(show_on_input=False, continuous_drawing=False, color_format=ColorFormat.BGR)
        self.tracked_data = self.tracked_data.reshape((faces, 68, 2))
        '''

        if self.tracked_data is None:
            self.tracked_data = np.zeros((68, 2))
            return None

        self.tracked_data = self.tracked_data[0]

        return self.tracked_data
