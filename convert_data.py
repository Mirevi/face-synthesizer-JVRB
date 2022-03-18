# -*- coding: utf-8 -*-
"""
This script converts given captured data into a face dataset.
"""
import os
import traceback
from math import sin, radians, floor
from shutil import rmtree

import numpy as np
import cv2 as cv
from numpy import uint8, uint16, float64
from sklearn.decomposition import PCA
from tqdm import tqdm
from config import ConfigType, new_argparse_config, BaseConfig, StandardConfig
from config.convert_config import ConvertConfigProvider
from data.preprocessing.depthFilling import new_depth_filler_instance, depth_filler_class, DepthFillingCOP
from tracking.eye_tracking import new_eye_tracker_instance, EyeTrackerInput
from tracking.face_aabbox_tracking import new_face_aabbox_tracker_instance, FaceAABBoxTrackerInput, \
    FaceAABBoxTrackingMethod
from tracking.face_tracking import new_face_tracker_instance, FaceTrackerInput
from util import count_images, create_text_file, read_text_file, mkdir, get_image_list
from util.image_tools import ColorFormat, visualize_images, get_image_excerpt, clip_16_bit_to_8_bit_uint


class ConvertData:
    def __init__(self, config: BaseConfig):
        # validate config
        if ConvertConfigProvider not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # set necessary members from config
        self.name = config['name']
        input_dir = os.path.join(config['input_root'], self.name)
        output_dir = os.path.join(config['output_root'], self.name)
        self.continue_process = config['continue_process']
        self.overwrite = config['overwrite']
        self.visualize = config['visualize']
        self.fill_depth_images = not config['no_depth_filling']
        self.horizontal_fov = config['horizontal_fov']
        self.output_image_size = config['output_image_size']
        self.padding = config['padding']
        self.depth_padding = config['depth_padding']
        self.face_depth = config['face_depth']

        # validate
        if self.continue_process and self.overwrite:
            raise ValueError('Cant continue process and overwrite simultaneously!')

        # define locations
        self.train_directories = self.get_directories(input_dir, output_dir, 'train')
        self.eval_directories = self.get_directories(input_dir, output_dir, 'eval')

        # check which input data exists
        self.train_input_exists = os.path.exists(self.train_directories['input_color'])
        self.eval_input_exists = os.path.exists(self.eval_directories['input_color'])

        # if overwrite then delete current output content
        if self.overwrite:
            rmtree(output_dir, ignore_errors=True)

        # check for existing output data
        if os.path.exists(output_dir) and not self.continue_process:
            raise RuntimeError(
                "A dataset already exists at location '{}'. To overwrite the existing dataset set "
                "the overwrite option in the config".format(output_dir))

        # get depth filler config
        depth_filler_algorithm_class = depth_filler_class(config['df_algorithm'])
        depth_filler_options = config.get_options_from_provider(depth_filler_algorithm_class)
        depth_filler_options.update(config.get_options_from_package(DepthFillingCOP))
        self.depth_filler_config = StandardConfig(depth_filler_options)
        self.depth_filler_config.add_option_package(DepthFillingCOP)
        self.depth_filler_config.add_package_provider(depth_filler_algorithm_class)

        # validate and prepare locations
        if self.train_input_exists:
            print('Detected train input data')
            self.validate_input_locations(self.train_directories)
            self.prepare_output_locations(self.train_directories)
            self.prepare_temporary_locations(self.train_directories)
            if self.fill_depth_images:
                self.prepare_filled_depth(self.train_directories, depth_filler_algorithm_class)
        if self.eval_input_exists:
            print('Detected eval input data')
            self.validate_input_locations(self.eval_directories)
            self.prepare_output_locations(self.eval_directories)
            self.prepare_temporary_locations(self.eval_directories)
            if self.fill_depth_images:
                self.prepare_filled_depth(self.eval_directories, depth_filler_algorithm_class)

        # declare or create necessary members
        self.image_color = None
        self.image_depth = None
        self.image_infrared = None
        self.image_feature = None
        self.image_eye_tracking = None
        self.image_color_scaled = None
        self.image_depth_scaled = None
        self.face_tracker = new_face_tracker_instance(config['face_tracking_method'], config)
        self.eye_tracker = new_eye_tracker_instance(config['eye_tracking_method'], config)
        self.multiple_face_detector = new_face_aabbox_tracker_instance(FaceAABBoxTrackingMethod.SynergyNet, config)
        self.depth_hole_filler = new_depth_filler_instance(config['df_algorithm'], config)
        self.df_image_size = config['df_image_size']
        self.df_padding = config['df_padding']
        self.df_patch_half_size = self.depth_hole_filler.get_max_patch_half_size()
        self.landmarks = np.zeros((70, 2), float)
        self.face_bounding_box = {}
        self.face_bounding_box_scaled = None
        self.desired_depth = None
        # triangle calculations for depth normalization
        alpha = self.horizontal_fov / 2
        beta = 90 - alpha
        self.triangle_sinus_division_term = sin(radians(alpha)) / sin(radians(beta))
        self.depth_scale_factor = None
        self.max_face_bb_size = {'width': 0, 'height': 0}
        self.general_scale_factor = None
        # pca
        pca_image_components = config['pca_image_components']
        pca_image_components = pca_image_components if pca_image_components <= 1 else int(pca_image_components)
        self.pca_image = PCA(pca_image_components)
        pca_landmarks_components = config['pca_landmarks_components']
        pca_landmarks_components = pca_landmarks_components if pca_landmarks_components <= 1 else int(pca_landmarks_components)
        self.pca_landmarks = PCA(pca_landmarks_components)

        # save config
        config.save_to_disk(output_dir, 'ConvertConfig.txt')

    def get_directories(self, input_dir, output_dir, subdir_name):
        return {
            'input': os.path.join(input_dir, subdir_name),
            'input_color': os.path.join(input_dir, subdir_name, 'Color'),
            'input_depth': os.path.join(input_dir, subdir_name, 'Depth'),
            'input_infrared': os.path.join(input_dir, subdir_name, 'Infrared'),
            'filled_depth': os.path.join(input_dir, subdir_name, 'Filled_Depth'),
            'temp': os.path.join(output_dir, subdir_name, 'Temp'),
            'temp_metadata': os.path.join(output_dir, subdir_name, 'Temp', 'Metadata'),
            'output': os.path.join(output_dir, subdir_name),
            'output_color': os.path.join(output_dir, subdir_name, 'Color'),
            'output_depth': os.path.join(output_dir, subdir_name, 'Depth'),
            'output_feature': os.path.join(output_dir, subdir_name, 'Feature'),
            'output_landmark': os.path.join(output_dir, subdir_name, 'Landmark'),
        }

    def validate_input_locations(self, directories):
        for key, value in directories.items():
            if key.startswith('input') and not os.path.exists(value):
                raise IOError("The input location '{}' does not exist: {}".format(key, value))

    def prepare_output_locations(self, directories):
        # create output dirs
        for key, value in directories.items():
            if key.startswith('output'):
                mkdir(value)

    def prepare_temporary_locations(self, directories):
        # create temp dirs
        for key, value in directories.items():
            if key.startswith('temp'):
                mkdir(value)

    def prepare_filled_depth(self, directories, depth_filler_class):
        filled_depth_dir = directories['filled_depth']
        filled_depth_config_file = os.path.join(filled_depth_dir, 'FilledDepthConfig.txt')
        depth_filler_config_dict = StandardConfig.string_to_dict(str(self.depth_filler_config))

        if os.path.exists(filled_depth_config_file):
            # compare config dicts
            file_reader = open(filled_depth_config_file, 'r')
            config_string = file_reader.read()
            file_reader.close()
            filled_depth_config_dict = StandardConfig.string_to_dict(config_string)

            # return if same settings
            if filled_depth_config_dict == depth_filler_config_dict:
                return
            else:
                print('Filled Depth location: {}'.format(filled_depth_dir))
                print('Existing filled Depth was created with different settings. '
                      'Deleting existing filled Depth and recreate all filled Depth during preparing for conversion.')

        # clear all filled depth data
        rmtree(filled_depth_dir, ignore_errors=True)
        mkdir(filled_depth_dir)

        # save current depth filler config
        depth_filler_config = StandardConfig(depth_filler_config_dict)
        depth_filler_config.add_option_package(DepthFillingCOP)
        depth_filler_config.add_package_provider(depth_filler_class)
        depth_filler_config.save_to_disk(filled_depth_dir, 'FilledDepthConfig.txt')

    def __call__(self):
        # prepare normalization
        if self.train_input_exists:
            self.prepare_conversion(self.train_directories, 'Prepare train data')
        if self.eval_input_exists:
            self.prepare_conversion(self.eval_directories, 'Prepare eval data')

        print('Compute general scale Factor')
        self.compute_general_scale_factor()

        # convert and normalize
        if self.train_input_exists:
            self.convert(self.train_directories, 'Convert train data')
        if self.eval_input_exists:
            self.convert(self.eval_directories, 'Convert eval data')

        # remove temporary dirs
        self.remove_temporary_locations(self.train_directories)
        self.remove_temporary_locations(self.eval_directories)

    def prepare_conversion(self, directories, tqdm_description=''):
        """
        Estimates Landmarks, computes axis aligned face bounding box, fills depth holes, sets the desired depth,
        computes the depth normalization scale factor, saves temporary metadata and filled depth and
        determines the maximum size of the axis aligned face bounding box.
        """
        input_files = self.get_input_files(directories)
        input_files_metadata = self.get_input_files(directories, consider_metadata=True)
        if self.continue_process and len(input_files_metadata['names']) > 1:
            latest_metadata_name = input_files_metadata['names'][-1]
            initial_index = max(0, input_files['names'].index(latest_metadata_name) - 1)  # -1 to avoid incomplete data
            self.load_data_from_already_prepared_data(directories, input_files_metadata)
        else:
            initial_index = 0

        # prepare remaining data
        pbar = tqdm(range(initial_index, len(input_files['names'])))
        pbar.set_description(tqdm_description, refresh=False)
        for i in pbar:
            try:
                pbar.set_postfix_str(self.get_tqdm_postfix(input_files, i))
                self.load_input_data_pair(directories, input_files, i)
                self.prepare_landmarks_and_bounding_box()
                self.prepare_depth(input_files, i)
                self.save_prepared_data(directories, input_files, i)
                self.update_maximum_face_bb_size()

                if self.visualize:
                    self.visualize_prepare_conversion()
            except Exception as e:
                #traceback.print_tb(e.__traceback__)
                print("Exception during preparing. Skipping image '{}'. "
                      "Exception message: {}".format(input_files['names'][i], e))
                continue

    def load_data_from_already_prepared_data(self, directories, input_files):
        pbar = tqdm(range(0, len(input_files['names'])))
        pbar.set_description('collect existing data', refresh=False)
        for i in pbar:
            pbar.set_postfix_str(self.get_tqdm_postfix(input_files, i))
            self.get_metadata_from_temp(directories, input_files, i)

            # set desired depth if is None
            if self.desired_depth is None:
                file_name_image = input_files['names'][i] + input_files['image_extensions'][i]
                if self.fill_depth_images and not input_files['filled_depth_exists'][i]:
                    self.load_input_data_pair(directories, input_files, i)
                    self.fill_depth_holes()
                    self.save_filled_depth_image(directories, input_files, i)
                else:
                    depth_dir_label = 'filled_depth' if self.fill_depth_images else 'input_depth'
                    self.load_input_image_depth(directories, depth_dir_label, file_name_image)
                self.desired_depth = self.get_nose_minimum_depth()

            # bb max size
            self.update_maximum_face_bb_size()

    def prepare_landmarks_and_bounding_box(self):
        # check for multiple faces
        face_aabbox_tracker_input = FaceAABBoxTrackerInput(self.image_color)
        bounding_box = self.multiple_face_detector.track_face_aabbox(face_aabbox_tracker_input)
        if bounding_box is None:
            raise RuntimeError('Invalid amount of faces in image.')

        # get landmarks
        self.estimate_landmarks()

        # get face bounding box
        self.face_bounding_box = self.get_landmark_bounding_box(self.landmarks)
        if self.is_bounding_box_invalid():
            raise ValueError('Face Bounding Box is invalid ({}). Skipping the image ...'.format(self.face_bounding_box))

    def estimate_landmarks(self):
        # get face landmarks
        face_tracker_input = FaceTrackerInput(self.image_color)
        landmarks_face = self.face_tracker.track_landmarks(face_tracker_input)

        # calculate individual eye tracking bounding boxes
        landmarks_left_eye = landmarks_face[36:41, :]
        landmarks_right_eye = landmarks_face[42:47, :]

        bbox_eye_left = self.get_landmark_bounding_box(landmarks_left_eye)
        bbox_eye_right = self.get_landmark_bounding_box(landmarks_right_eye)

        # get eye landmarks
        eye_tracker_input = EyeTrackerInput(self.image_color.astype(np.uint8), self.image_infrared.astype(np.uint8),
                                            bbox_eye_left, bbox_eye_right)
        landmarks_eyes = self.eye_tracker.track_landmarks(eye_tracker_input)

        # create eye tracking area visualization
        if self.visualize:
            self.image_eye_tracking = np.copy(self.image_color)
            cv.rectangle(self.image_eye_tracking, (int(bbox_eye_left['x']), int(bbox_eye_left['y'])),
                         (int(bbox_eye_left['x'] + bbox_eye_left['width']),
                          int(bbox_eye_left['y'] + bbox_eye_left['height'])),
                         (255, 0, 0), 6)
            cv.rectangle(self.image_eye_tracking, (int(bbox_eye_right['x']), int(bbox_eye_right['y'])),
                         (int(bbox_eye_right['x'] + bbox_eye_right['width']),
                          int(bbox_eye_right['y'] + bbox_eye_right['height'])),
                         (0, 255, 0), 6)
            cv.circle(self.image_eye_tracking, (int(landmarks_eyes[0, 0]), int(landmarks_eyes[0, 1])), 5, (0, 0, 255),
                      -1)
            cv.circle(self.image_eye_tracking, (int(landmarks_eyes[1, 0]), int(landmarks_eyes[1, 1])), 5, (0, 0, 255),
                      -1)

        # concatenate landmarks
        self.landmarks = np.concatenate((landmarks_face, landmarks_eyes), axis=0)

    def is_bounding_box_invalid(self):
        if self.face_bounding_box is None:
            return True  # No face or too many faces detected
        # check boundaries
        elif self.face_bounding_box['x'] < 0 or self.face_bounding_box['y'] < 0 \
                or self.face_bounding_box['x'] + self.face_bounding_box['width'] > self.image_color.shape[0] \
                or self.face_bounding_box['y'] + self.face_bounding_box['height'] > self.image_color.shape[1]:
            return True

        return False

    @staticmethod
    def get_landmark_bounding_box(landmarks):
        x1 = np.amin(landmarks[:, 0])
        y1 = np.amin(landmarks[:, 1])
        x2 = np.amax(landmarks[:, 0])
        y2 = np.amax(landmarks[:, 1])
        return {'x': x1, 'y': y1, 'width': x2 - x1, 'height': y2 - y1}

    def prepare_depth(self, input_files, index):
        """
        fills depth holes, sets the desired depth and computes the depth normalization scale factor
        """
        # fill depth
        if self.fill_depth_images and not input_files['filled_depth_exists'][index]:
            self.fill_depth_holes()

        actual_depth = self.get_nose_minimum_depth()

        # set desired depth if is None
        if self.desired_depth is None:
            self.desired_depth = actual_depth

        # compute depth scale factor
        self.compute_depth_normalization_scale_factor(actual_depth)

    def fill_depth_holes(self):
        """
        Fills the holes in the depth image. It is important to call this method before scaling the image because
        otherwise the interpolation method of the scaling method creates unreal values e.g. if you have 2 Pixels
        (100, 0) and scale them to 3 Pixels you get (100, 0) -> (100, 50, 0) if you are using linear interpolation.
        """
        padding = self.df_padding + self.df_patch_half_size
        image_size = (self.df_image_size, self.df_image_size)

        # crop images
        image_color_excerpt = get_image_excerpt(self.image_color, self.face_bounding_box, padding)
        image_depth_excerpt = get_image_excerpt(self.image_depth, self.face_bounding_box, padding, return_copy=False)

        # scale images
        color = cv.resize(image_color_excerpt, image_size, interpolation=cv.INTER_NEAREST)
        depth = cv.resize(image_depth_excerpt, image_size, interpolation=cv.INTER_NEAREST)

        # fill holes
        depth = self.depth_hole_filler(depth, color)

        # rescale depth image
        depth = cv.resize(depth, (image_depth_excerpt.shape[1], image_depth_excerpt.shape[0]),
                          interpolation=cv.INTER_LINEAR)
        depth = np.reshape(depth, (image_depth_excerpt.shape[0], image_depth_excerpt.shape[1]))

        # fill into original image (by reference)
        image_depth_excerpt[:, :] = np.where(image_depth_excerpt > 0, image_depth_excerpt, depth)

    def get_nose_minimum_depth(self):
        # determine nose point area
        nose_area = self.get_nose_area()

        # get nose point depth value
        nose_area_depth_values = get_image_excerpt(self.image_depth, nose_area)
        non_zero_indices = np.nonzero(nose_area_depth_values)
        if len(non_zero_indices) == 0:
            raise ValueError('Nose Area has no depth value which is greater than 0! Skipping image ...')
        actual_depth = np.amin(nose_area_depth_values[non_zero_indices])

        return actual_depth

    def compute_depth_normalization_scale_factor(self, actual_depth):
        a = float(actual_depth) * self.triangle_sinus_division_term
        a_hat = float(self.desired_depth) * self.triangle_sinus_division_term
        self.depth_scale_factor = a / a_hat

    def save_prepared_data(self, directories, input_files, index):
        self.save_metadata(directories, index, input_files)

        if self.fill_depth_images and not input_files['filled_depth_exists'][index]:
            self.save_filled_depth_image(directories, input_files, index)

    def save_metadata(self, directories, index, input_files):
        file_name = '{}.txt'.format(input_files['names'][index])
        file_path = os.path.join(directories['temp_metadata'], file_name)
        content = ["depth_scale_factor: {}\n".format(self.depth_scale_factor)]
        content.extend(self.get_landmarks_file_content())
        create_text_file(file_path, content)

    def save_filled_depth_image(self, directories, input_files, index):
        # save depth image in filled depth dir
        file_name = input_files['names'][index] + input_files['image_extensions'][index]
        image_depth_file_path = os.path.join(directories['filled_depth'], file_name)
        cv.imwrite(image_depth_file_path, self.image_depth.astype(uint16))

    def update_maximum_face_bb_size(self):
        # apply scale factor to face_bounding_box
        depth_scaled_face_bounding_box = {x: y * self.depth_scale_factor for x, y in
                                          self.face_bounding_box.items()}
        # update max face bounding box size
        self.max_face_bb_size = {
            'width': max(depth_scaled_face_bounding_box["width"], self.max_face_bb_size['width']),
            'height': max(depth_scaled_face_bounding_box["height"], self.max_face_bb_size['height']),
        }

    def compute_general_scale_factor(self):
        # get scale factor
        print('max face bounding box size: ({}, {})'.format(self.max_face_bb_size['width'],
                                                            self.max_face_bb_size['height']))
        scale_factor_x = (self.output_image_size - 2 * self.padding) / self.max_face_bb_size["width"]
        scale_factor_y = (self.output_image_size - 2 * self.padding) / self.max_face_bb_size["height"]
        self.general_scale_factor = min(scale_factor_x, scale_factor_y)
        print('scale factor: {}'.format(self.general_scale_factor))

    def visualize_prepare_conversion(self):
        # bounding box
        image_bounding_box = np.copy(self.image_color).astype(uint8)
        cv.rectangle(image_bounding_box, (int(self.face_bounding_box['x']), int(self.face_bounding_box['y'])),
                     (int(self.face_bounding_box['x'] + self.face_bounding_box['width']),
                      int(self.face_bounding_box['y'] + self.face_bounding_box['height'])), (0, 255, 0), 4)
        # landmarks
        image_landmarks = np.copy(self.image_color)
        for landmark in self.landmarks:
            cv.circle(image_landmarks, (int(landmark[0]), int(landmark[1])), 4, (0, 255, 0), -1)
        image_landmarks = get_image_excerpt(image_landmarks, self.face_bounding_box, self.padding).astype(uint8)

        # nose area
        nose_area = self.get_nose_area()
        image_nose_area = np.copy(self.image_depth)
        cv.rectangle(image_nose_area, (int(nose_area['x']), int(nose_area['y'])),
                     (int(nose_area['x'] + nose_area['width']), int(nose_area['y'] + nose_area['height'])), 0, 4)
        image_nose_area = get_image_excerpt(image_nose_area, self.face_bounding_box, self.padding).astype(uint8)

        # eye tracking
        self.image_eye_tracking = get_image_excerpt(self.image_eye_tracking, self.face_bounding_box,
                                                    self.padding).astype(uint8)

        visualize_images(
            [image_bounding_box, image_landmarks, image_nose_area, self.image_eye_tracking],
            ['face bounding box', 'landmarks', 'nose area', 'eye_tracking'], ColorFormat.BGR)

    def convert(self, directories, tqdm_description=''):
        input_files = self.get_input_files(directories, consider_metadata=True)

        if self.continue_process:
            initial_index = max(0, count_images(directories['output_color']) - 1)  # -1 to avoid incomplete data
        else:
            initial_index = 0

        pbar = tqdm(range(initial_index, len(input_files['names'])))
        pbar.set_description(tqdm_description, refresh=False)
        for i in pbar:
            postfix_str = self.get_tqdm_postfix(input_files, i)
            pbar.set_postfix_str(postfix_str)
            self.load_input_data_pair(directories, input_files, i)
            self.get_metadata_from_temp(directories, input_files, i)

            self.scale_and_crop()
            self.normalize_depth()
            self.apply_pca()
            self.compute_feature_image()

            self.save_output_data(directories, input_files, i)

            if self.visualize:
                self.visualize_convert()

    def get_input_files(self, directories, consider_metadata=False):
        files_image_color = get_image_list(directories['input_color'])
        files_image_depth = get_image_list(directories['input_depth'])
        files_image_infrared = get_image_list(directories['input_infrared'])
        files_metadata = [i for i in os.listdir(directories['temp_metadata']) if i.lower().endswith('.txt')]

        # sort
        files_image_color.sort(key=len)
        files_image_depth.sort(key=len)
        files_image_infrared.sort(key=len)
        files_metadata.sort(key=len)

        # validate images
        self.validate_image_files_lists(files_image_color, files_image_depth, files_image_infrared)

        # get file names and extensions
        splitted_image_files = [os.path.splitext(file) for file in files_image_color]
        splitted_metadata_files = [os.path.splitext(file) for file in files_metadata]
        metadata_file_names = [file[0] for file in splitted_metadata_files]

        if consider_metadata:
            names = [file[0] for file in splitted_image_files if file[0] in metadata_file_names]
            image_extensions = [file[1] for file in splitted_image_files if file[0] in metadata_file_names]
        else:
            names = [file[0] for file in splitted_image_files]
            image_extensions = [file[1] for file in splitted_image_files]

        input_files = {
            'names': names,
            'image_extensions': image_extensions,  # list for multiple extension support
        }

        if self.fill_depth_images:
            filled_depth_paths = [
                os.path.join(directories['filled_depth'], input_files['names'][i] + input_files['image_extensions'][i])
                for i in range(len(names))]
            filled_depth_exists = [True if os.path.exists(file_path) else False for file_path in filled_depth_paths]
            input_files['filled_depth_exists'] = filled_depth_exists

        return input_files

    def validate_image_files_lists(self, files_color, files_depth, files_infrared):
        # validate amount of files
        if not len(files_color) == len(files_depth) == len(files_infrared):
            raise IOError("Different Amount of Train Files. "
                          "Color: {}, Depth {}, Infrared: {}".format(len(files_color),
                                                                     len(files_depth),
                                                                     len(files_infrared)))

        # validate same file names
        for i in range(len(files_color)):
            if not files_color[i] == files_depth[i] == files_infrared[i]:
                raise IOError("The Data in the subdirectories of the Train input dir have different names! "
                              + "Color: {}, Depth: {}, Infrared: {}".format(files_color[i],
                                                                            files_depth[i],
                                                                            files_infrared[i]))

    def get_tqdm_postfix(self, input_files, index):
        postfix_str = {'file name': input_files['names'][index]}
        if self.fill_depth_images:
            postfix_str['filled depth exists'] = input_files['filled_depth_exists'][index]
        return postfix_str

    def load_input_data_pair(self, directories, input_files, index):
        depth_dir_label = 'input_depth'
        if self.fill_depth_images and input_files['filled_depth_exists'][index]:
            depth_dir_label = 'filled_depth'

        # define full file names
        file_name_image = input_files['names'][index] + input_files['image_extensions'][index]

        self.load_input_image_color(directories, file_name_image)
        self.load_input_image_depth(directories, depth_dir_label, file_name_image)
        self.load_input_image_infrared(directories, file_name_image)

        # validate shapes
        if not self.image_color.shape[0:2] == self.image_depth.shape[0:2] == self.image_infrared.shape[0:2]:
            raise RuntimeError("Shapes of input images with filename '{}' are different!"
                               "color: {}, depth: {}, infrared: {}".format(file_name_image,
                                                                           self.image_color.shape,
                                                                           self.image_depth.shape,
                                                                           self.image_infrared.shape))

    def load_input_image_color(self, directories, file_name_image):
        image_color_path = os.path.join(directories['input_color'], file_name_image)
        self.image_color = cv.imread(image_color_path, cv.IMREAD_COLOR)
        self.image_color = np.asarray(self.image_color).astype(float)
        self.image_color = self.image_color[:, :, 0:3]

    def load_input_image_depth(self, directories, depth_dir_label, file_name_image):
        image_depth_path = os.path.join(directories[depth_dir_label], file_name_image)
        self.image_depth = cv.imread(image_depth_path, cv.IMREAD_UNCHANGED)
        self.image_depth = np.asarray(self.image_depth).astype(float)
        shape_depth = self.image_depth.shape
        self.image_depth = np.reshape(self.image_depth, (shape_depth[0], shape_depth[1]))

    def load_input_image_infrared(self, directories, file_name_image):
        image_infrared_path = os.path.join(directories['input_infrared'], file_name_image)
        self.image_infrared = cv.imread(image_infrared_path, cv.IMREAD_GRAYSCALE)
        self.image_infrared = np.asarray(self.image_infrared)
        shape_infrared = self.image_infrared.shape
        self.image_infrared = np.reshape(self.image_infrared, (shape_infrared[0], shape_infrared[1]))

    def get_metadata_from_temp(self, directories, input_files, index):
        file_path = os.path.join(directories['temp_metadata'], '{}.txt'.format(input_files['names'][index]))
        content = read_text_file(file_path)

        for line in content:
            if line.startswith('depth_scale_factor'):
                splitted = line.split(': ')
                self.depth_scale_factor = float(splitted[1])

            elif line.startswith('aabb_x'):
                splitted = line.split(': ')
                self.face_bounding_box['x'] = float(splitted[1])

            elif line.startswith('aabb_y'):
                splitted = line.split(': ')
                self.face_bounding_box['y'] = float(splitted[1])

            elif line.startswith('aabb_width'):
                splitted = line.split(': ')
                self.face_bounding_box['width'] = float(splitted[1])

            elif line.startswith('aabb_height'):
                splitted = line.split(': ')
                self.face_bounding_box['height'] = float(splitted[1])

            elif line.startswith('X'):
                splitted = line.split(': ')
                splitted = splitted[1].split(',')[:-1]
                if self.landmarks.shape[0] != len(splitted):
                    self.landmarks = np.zeros((len(splitted), 2), float)
                self.landmarks[:, 0] = [float(x) for x in splitted]

            elif line.startswith('Y'):
                splitted = line.split(': ')
                splitted = splitted[1].split(',')[:-1]
                if self.landmarks.shape[0] != len(splitted):
                    self.landmarks = np.zeros((len(splitted), 2), float)
                self.landmarks[:, 1] = [float(x) for x in splitted]

    def scale_and_crop(self):
        # apply depth scale factor and scale factor
        scale_factor = self.depth_scale_factor * self.general_scale_factor
        self.apply_scale_factor(scale_factor)

        # compute and apply crop region
        self.compute_crop_region()
        self.apply_crop_region()

    def apply_scale_factor(self, scale_factor):
        self.image_color = cv.resize(self.image_color, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv.INTER_CUBIC)
        self.image_depth = cv.resize(self.image_depth, None, fx=scale_factor, fy=scale_factor,
                                     interpolation=cv.INTER_CUBIC)
        self.landmarks *= scale_factor
        self.face_bounding_box.update((x, y * scale_factor) for x, y in self.face_bounding_box.items())

        if self.visualize:
            self.image_color_scaled = np.copy(self.image_color)
            self.image_depth_scaled = np.copy(self.image_depth)
            self.face_bounding_box_scaled = self.face_bounding_box.copy()

    def compute_crop_region(self):
        face_bounding_box_center = {
            "x": self.face_bounding_box["x"] + self.face_bounding_box['width'] / 2,
            "y": self.face_bounding_box["y"] + self.face_bounding_box['height'] / 2,
        }
        self.crop_region = {
            "x": int(round(face_bounding_box_center["x"] - self.output_image_size / 2)),
            "y": int(round(face_bounding_box_center["y"] - self.output_image_size / 2)),
            "width": self.output_image_size,
            "height": self.output_image_size,
        }

    def apply_crop_region(self):
        # crop images
        self.image_color = get_image_excerpt(self.image_color, self.crop_region)
        self.image_depth = get_image_excerpt(self.image_depth, self.crop_region)
        # crop landmarks
        self.landmarks[:, 0] -= self.crop_region["x"]
        self.landmarks[:, 1] -= self.crop_region["y"]
        # crop aabb
        self.face_bounding_box['x'] -= self.crop_region["x"]
        self.face_bounding_box['y'] -= self.crop_region["y"]

    def normalize_depth(self):
        # add depth delta to image_depth values
        nose_area = self.get_nose_area()
        actual_depth = np.amin(get_image_excerpt(self.image_depth, nose_area))
        self.image_depth += self.desired_depth - actual_depth

        # convert 16 bit depth to 8 bit depth
        self.image_depth = clip_16_bit_to_8_bit_uint(self.image_depth,
                                                     int(self.desired_depth - self.depth_padding),
                                                     int(self.desired_depth + self.face_depth))

        # apply erode to depth image to remove noise at edges
        kernel_size = floor(self.depth_scale_factor * self.general_scale_factor)
        kernel_size = kernel_size * 2 + 1
        kernel_size = (kernel_size, kernel_size)
        image_erode_mask = np.where(self.image_depth > 0, 255, 0).astype(uint8)
        image_erode_mask = cv.erode(image_erode_mask, cv.getStructuringElement(cv.MORPH_RECT, kernel_size))
        self.image_depth = cv.bitwise_and(self.image_depth, image_erode_mask)

    def get_nose_area(self):
        nose_area = {'x': self.landmarks[32, 0], 'y': self.landmarks[29, 1],
                     'width': self.landmarks[34, 0] - self.landmarks[32, 0],
                     'height': self.landmarks[32, 1] - self.landmarks[29, 1]}
        return nose_area

    def apply_pca(self):
        self.apply_pca_image_compression()
        self.apply_pca_landmarks_compression()

    def apply_pca_image_compression(self):
        color_data_type = self.image_color.dtype
        depth_data_type = self.image_depth.dtype
        b = self.image_color[:, :, 0]
        g = self.image_color[:, :, 1]
        r = self.image_color[:, :, 2]
        d = self.image_depth

        transformed_b = self.pca_image.fit_transform(b)
        inverse_b = self.pca_image.inverse_transform(transformed_b)
        transformed_g = self.pca_image.fit_transform(g)
        inverse_g = self.pca_image.inverse_transform(transformed_g)
        transformed_r = self.pca_image.fit_transform(r)
        inverse_r = self.pca_image.inverse_transform(transformed_r)
        transformed_d = self.pca_image.fit_transform(d)
        inverse_d = self.pca_image.inverse_transform(transformed_d)

        self.image_color[:, :, 0] = inverse_b[:, :]
        self.image_color[:, :, 1] = inverse_g[:, :]
        self.image_color[:, :, 2] = inverse_r[:, :]
        self.image_depth = inverse_d

        self.image_color = self.image_color.astype(color_data_type)
        self.image_depth = self.image_depth.astype(depth_data_type)

    def apply_pca_landmarks_compression(self):
        self.landmarks = self.landmarks.transpose((1, 0))

        transformed_landmarks = self.pca_landmarks.fit_transform(self.landmarks)
        self.landmarks = self.pca_landmarks.inverse_transform(transformed_landmarks)

        self.landmarks = self.landmarks.transpose((1, 0))

    def compute_feature_image(self):
        self.image_feature = np.zeros((self.output_image_size, self.output_image_size, 1))

        for i in range(1, self.landmarks.shape[0]):
            if i == 17 or i == 22 or i == 27 or i == 31:
                continue

            prev = i - 1
            if i == 36:
                prev = 41
            elif i == 42:
                prev = 47
            elif i == 48:
                prev = 59
            elif i == 60:
                prev = 67

            x1 = int(round(self.landmarks[prev][0]))
            y1 = int(round(self.landmarks[prev][1]))
            x2 = int(round(self.landmarks[i][0]))
            y2 = int(round(self.landmarks[i][1]))

            if i == 68 or i == 69:
                cv.circle(self.image_feature, (x2, y2), int(2), 255, 4)
            else:
                cv.line(self.image_feature, (x1, y1), (x2, y2), 255, 4)

        self.image_feature = np.asarray(self.image_feature).astype(float)

    def save_output_data(self, directories, input_files, index):
        file_name = input_files['names'][index] + input_files['image_extensions'][index]

        # define file paths
        image_color_path = os.path.join(directories['output_color'], file_name)
        image_depth_path = os.path.join(directories['output_depth'], file_name)
        image_feature_path = os.path.join(directories['output_feature'], file_name)

        # export images (files have the same names)
        cv.imwrite(image_color_path, self.image_color)
        cv.imwrite(image_depth_path, self.image_depth)
        cv.imwrite(image_feature_path, self.image_feature)

        # save landmarks
        file_path = os.path.join(directories['output_landmark'], '{}.txt'.format(input_files['names'][index]))
        content = self.get_landmarks_file_content()
        create_text_file(file_path, content)

    def get_landmarks_file_content(self):
        # landmarks to ',' separated strings
        landmarks_x = ''
        landmarks_y = ''
        landmarks = np.around(self.landmarks, decimals=4)
        for current in landmarks:
            landmarks_x += "{},".format(current[0])
            landmarks_y += "{},".format(current[1])

        content = [
            "aabb_x: {}\n".format(self.face_bounding_box['x']),
            "aabb_y: {}\n".format(self.face_bounding_box['y']),
            "aabb_width: {}\n".format(self.face_bounding_box['width']),
            "aabb_height: {}\n".format(self.face_bounding_box['height']),
            "X: {}\n".format(landmarks_x),
            "Y: {}".format(landmarks_y),
        ]

        return content

    def visualize_convert(self):
        # crop area
        image_crop_area = np.copy(self.image_color_scaled).astype(uint8)
        cv.rectangle(image_crop_area, (int(self.crop_region['x']), int(self.crop_region['y'])),
                     (int(self.crop_region['x'] + self.crop_region['width']),
                      int(self.crop_region['y'] + self.crop_region['height'])), (0, 255, 0), 4)
        cv.rectangle(image_crop_area,
                     (int(self.face_bounding_box_scaled['x']), int(self.face_bounding_box_scaled['y'])),
                     (int(self.face_bounding_box_scaled['x'] + self.face_bounding_box_scaled['width']),
                      int(self.face_bounding_box_scaled['y'] + self.face_bounding_box_scaled['height'])),
                     (0, 255, 255), 4)
        # landmarks
        image_landmarks = np.copy(self.image_color).astype(uint8)
        for landmark in self.landmarks:
            cv.circle(image_landmarks, (int(landmark[0]), int(landmark[1])), 4, (0, 255, 0), -1)

        visualize_images(
            [image_crop_area, self.image_depth, self.image_feature, image_landmarks],
            ['crop area', 'depth', 'feature', 'landmarks'], ColorFormat.BGR)

    def remove_temporary_locations(self, directories):
        for key, value in directories.items():
            if key.startswith('temp'):
                rmtree(value, ignore_errors=True)


if __name__ == "__main__":
    # get configuration
    script_config = new_argparse_config(ConfigType.Convert)
    script_config.gather_options()
    script_config.print()

    convert_data_script = ConvertData(script_config)
    convert_data_script()
