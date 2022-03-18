# -*- coding: utf-8 -*-
"""
Captures and saves images from a k4a device which can then be used to create a dataset for AI training.
This Script saves rgb, depth and infrared images.
"""

import os
import sys
import time
from math import floor
from shutil import rmtree

import numpy as np
import cv2 as cv
from tqdm import tqdm
from pyk4a import PyK4A, Config, ColorResolution, ImageFormat, DepthMode, FPS, WiredSyncMode

from config import BaseConfig, new_argparse_config, ConfigType
from config.capture_config import CaptureConfigProvider
from tracking.face_aabbox_tracking import new_face_aabbox_tracker_instance, FaceAABBoxTrackerInput
from tracking.face_alignment_tracking import new_face_alignment_tracker_instance, FaceAlignmentTrackerInput
from util import mkdirs, ColorFormat, count_images, visualize_images_cv, visualize_images, get_image_excerpt, \
    convert_into_grayscale

def get_azure_config():
    return Config(
        color_resolution=ColorResolution.RES_1536P,
        color_format=ImageFormat.COLOR_BGRA32,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_30,
        synchronized_images_only=True,
        depth_delay_off_color_usec=0,
        wired_sync_mode=WiredSyncMode.STANDALONE,
        subordinate_delay_off_master_usec=0,
        disable_streaming_indicator=False,
    )

class CaptureData:
    @staticmethod
    def get_azure_config():
        return Config(
            color_resolution=ColorResolution.RES_3072P,
            color_format=ImageFormat.COLOR_BGRA32,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_15,
            synchronized_images_only=True,
            depth_delay_off_color_usec=0,
            wired_sync_mode=WiredSyncMode.STANDALONE,
            subordinate_delay_off_master_usec=0,
            disable_streaming_indicator=False,
        )

    @staticmethod
    def get_depth_illuminated_area(azure_config):
        if azure_config.color_resolution is ColorResolution.RES_3072P \
                and azure_config.depth_mode is DepthMode.NFOV_UNBINNED:
            return {
                'x': 1050,
                'y': 400,
                'width': 2950 - 1050,
                'height': 2670 - 400,
            }
        elif azure_config.color_resolution is ColorResolution.RES_2160P \
                and azure_config.depth_mode is DepthMode.NFOV_UNBINNED:
            return {
                'x': 900,
                'y': 170,
                'width': 2840 - 900,
                'height': 2020 - 170,
            }

        raise NotImplementedError('Given Azure Config is not supported!')

    def __init__(self, config: BaseConfig):
        # validate config
        if CaptureConfigProvider not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # get necessary values from config
        self.train_image_amount = config['train_image_amount']
        self.eval_image_amount = config['eval_image_amount']
        self.image_id_offset = config['image_id_offset']
        self.max_yaw = config['max_yaw']
        self.max_pitch = config['max_pitch']
        self.max_roll = config['max_roll']
        self.output_dir = os.path.join(config['output_root'], config['name'])
        self.overwrite = config['overwrite']
        self.continue_process = config['continue_process']
        self.visualize = config['visualize']

        # validation
        if not self.overwrite and not self.continue_process and os.path.exists(self.output_dir):
            raise RuntimeError("A capture already exists at location '{}'. Set the overwrite option of "
                               "the config to True if you want to replace the capture.".format(self.output_dir))
        if self.continue_process and self.overwrite:
            raise ValueError('Cant continue process and overwrite at the same Time!')

        # declare and initialize necessary locations
        self.output_dir_train_color = os.path.join(self.output_dir, 'train', 'Color')
        self.output_dir_train_depth = os.path.join(self.output_dir, 'train', 'Depth')
        self.output_dir_train_infrared = os.path.join(self.output_dir, 'train', 'Infrared')
        self.output_dir_eval_color = os.path.join(self.output_dir, 'eval', 'Color')
        self.output_dir_eval_depth = os.path.join(self.output_dir, 'eval', 'Depth')
        self.output_dir_eval_infrared = os.path.join(self.output_dir, 'eval', 'Infrared')

        # declare or create necessary members
        self.initial_index = 0
        if self.eval_image_amount != 0:
            self.eval_capture_rate = (self.train_image_amount + self.eval_image_amount) / self.eval_image_amount
        else:
            self.eval_capture_rate = sys.maxsize
        self.face_aabbox_tracker = new_face_aabbox_tracker_instance(config['face_aabbox_tracking_method'], config)
        self.face_alignment_tracker = new_face_alignment_tracker_instance(config['face_align_tracking_method'], config)

        # prepare output location
        # delete content if overwrite is true
        if self.overwrite:
            rmtree(self.output_dir, ignore_errors=True)

        # create directories if not exists
        if self.train_image_amount > 0:
            mkdirs([self.output_dir_train_color, self.output_dir_train_depth, self.output_dir_train_infrared])
        if self.eval_image_amount > 0:
            mkdirs([self.output_dir_eval_color, self.output_dir_eval_depth, self.output_dir_eval_infrared])

        # if continue process get initial index
        if self.continue_process:
            self.initial_index = count_images(self.output_dir_train_color) + count_images(
                self.output_dir_eval_color) - 1  # -1 to avoid incomplete data

        # save config
        config.save_to_disk(self.output_dir, 'CaptureConfig.txt')

    def __call__(self):
        # Load and start camera
        azure_config = self.get_azure_config()
        azure = PyK4A(config=azure_config)
        azure.start()

        illuminated_depth_area = self.get_depth_illuminated_area(azure_config)

        print('Starting to Capture Data. Initial Index: {} of {} (train) + {} (eval) = {} images.'.format(
            self.initial_index, self.train_image_amount, self.eval_image_amount,
            self.train_image_amount + self.eval_image_amount))
        i = 0
        accumulated_eval_rate = self.eval_capture_rate
        with tqdm(total=self.train_image_amount + self.eval_image_amount - self.initial_index, position=0,
                  leave=True) as pbar:
            while i in range(self.initial_index, self.train_image_amount + self.eval_image_amount):
                # to determine illuminated_depth_area
                '''capture = azure.get_capture()
                img_depth = capture.transformed_depth
                cv.rectangle(img_depth, (900, 170), (2840, 2020), 340, 5)
                visualize_images([img_depth], ["Depth"])
                continue'''

                # get capture
                capture = azure.get_capture()
                img_color = get_image_excerpt(capture.color[:, :, :3], illuminated_depth_area)
                img_depth = get_image_excerpt(capture.transformed_depth, illuminated_depth_area)
                img_infrared = get_image_excerpt(capture.transformed_ir, illuminated_depth_area)

                # visualize images
                if self.visualize:
                    visualize_images([img_color, img_depth, img_infrared], ["Color", "Depth", "Infrared"],
                                     ColorFormat.BGR)

                # get face bounding box
                face_aabbox_tracker_input = FaceAABBoxTrackerInput(img_color, landmarks=None)
                bounding_box = self.face_aabbox_tracker.track_face_aabbox(face_aabbox_tracker_input)
                if self.is_bounding_box_invalid(bounding_box, img_color.shape):
                    continue

                # check for unsuitable face alignment
                face_alignment_tracker_input = FaceAlignmentTrackerInput(img_color, bounding_box)
                angles = self.face_alignment_tracker.track_face_alignment(face_alignment_tracker_input)
                if self.are_angles_invalid(angles):
                    continue

                # determine file name
                image_files_name = '{}.png'.format(i + self.image_id_offset)

                # determine whether images are train or eval images
                are_eval_images = i + 1 == floor(accumulated_eval_rate)

                # update eval rate if are_eval_images
                if are_eval_images:
                    accumulated_eval_rate += self.eval_capture_rate

                # save images
                if are_eval_images:
                    cv.imwrite(os.path.join(self.output_dir_eval_color, image_files_name), img_color)
                    cv.imwrite(os.path.join(self.output_dir_eval_depth, image_files_name), img_depth)
                    cv.imwrite(os.path.join(self.output_dir_eval_infrared, image_files_name), img_infrared)
                else:
                    cv.imwrite(os.path.join(self.output_dir_train_color, image_files_name), img_color)
                    cv.imwrite(os.path.join(self.output_dir_train_depth, image_files_name), img_depth)
                    cv.imwrite(os.path.join(self.output_dir_train_infrared, image_files_name), img_infrared)

                # update while progress
                i += 1
                pbar.update()

    def is_bounding_box_invalid(self, bounding_box, img_shape):
        if bounding_box is None:
            return True  # No face or too many faces detected
        # check boundaries  TODO add minimum padding
        elif bounding_box['x'] < 0 or bounding_box['y'] < 0 \
                or bounding_box['x'] + bounding_box['width'] > img_shape[0] \
                or bounding_box['y'] + bounding_box['height'] > img_shape[1]:
            return True

        return False

    def are_angles_invalid(self, angles):
        if abs(angles['yaw']) > self.max_yaw \
                or abs(angles['pitch']) > self.max_pitch \
                or abs(angles['roll']) > self.max_roll:
            return True

        return False


if __name__ == "__main__":
    # get configuration
    script_config = new_argparse_config(ConfigType.Capture)
    script_config.gather_options()
    script_config.print()

    capture_data = CaptureData(script_config)
    capture_data()
