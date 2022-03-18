# -*- coding: utf-8 -*-
"""
This Script implements the basic 2D Landmark Tracker functionalities.
"""
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
from numpy import uint8

from tracking.base_tracker import BaseTracker
from util.image_tools import visualize_images, ColorFormat


class LandmarkTrackerInput:
    def __init__(self, image=None):
        self.image = image


class LandmarkTracker(BaseTracker, ABC):
    """
    A Landmark Tracker can track landmarks from an input image.
    """
    @abstractmethod
    def track_landmarks(self, input_data: LandmarkTrackerInput):
        self.input = input_data

    @staticmethod
    @abstractmethod
    def number_of_landmarks() -> int:
        return -1

    def show_latest_2d_landmarks(self, show_on_input=False, mark_landmarks: list = None,
                                 color_format=ColorFormat.RGB, continuous_drawing=True, pause_interval=3):
        image = self.input.image

        if mark_landmarks is None:
            mark_landmarks = []
        if image is None or self.tracked_data is None:
            print("There are no landmarks available to show.")
            return

        h, w, c = image.shape

        if show_on_input:
            landmarks_preview = np.copy(image).astype(uint8)
        else:
            landmarks_preview = np.zeros((h, w, 3), dtype=uint8)

        for lm in self.tracked_data:
            x = int(lm[0])
            y = int(lm[1])
            cv.circle(landmarks_preview, (x, y), 4, (0, 255, 0), -1)

        for i in mark_landmarks:
            if 0 <= i < len(self.tracked_data):
                lm = self.tracked_data[i]
                x = int(lm[0])
                y = int(lm[1])
                cv.circle(landmarks_preview, (x, y), 4, (0, 255, 255), -1)

        visualize_images([image.astype(uint8), landmarks_preview], ['Input', 'Landmarks'],
                         color_format, continuous_drawing, pause_interval)
