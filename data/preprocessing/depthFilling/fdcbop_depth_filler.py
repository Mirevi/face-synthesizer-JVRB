from argparse import ArgumentParser

import sys
import math
import random

import numpy as np
import cv2 as cv

from config import ConfigOptionPackage, ConfigOptionMetadata
from .depth_filling import DepthFiller
from util.image_tools import ColorFormat, visualize_images


class FDCBOPDepthFillerCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'df_patch_size', 17, 'The patch size used for depth filling.'),
            ConfigOptionMetadata(int, 'df_ext_patch_size', 29,
                                 'The extended patch size used to compute the reliability of similar patches.'),
            ConfigOptionMetadata(int, 'df_source_amount', 10000,
                                 'The amount of source region pixels used to compute the most similar patch. This value effects the performance and quality!'),
            ConfigOptionMetadata(int, 'df_blur_ksize', 9, 'blur kernel size'),
            ConfigOptionMetadata(bool, 'df_no_blur', False, 'Dont use blur at end of process.'),
            ConfigOptionMetadata(bool, 'df_visualize', False, 'Visualize the depth filling process.'),
            ConfigOptionMetadata(bool, 'df_verbose', False, 'Gives additional console output for debugging.'),
        ]


class FDCBOPDepthFiller(DepthFiller):
    """
    FDCBOP
    from https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.16.0116.0062#:~:text=The%20first%20hole%E2%80%90filling%20method,depth%20information%20is%20also%20proposed.
    III.
    """

    @staticmethod
    def get_required_option_packages() -> list:
        packages = super(FDCBOPDepthFiller, FDCBOPDepthFiller).get_required_option_packages()
        packages.append(FDCBOPDepthFillerCOP)
        return packages

    def __init__(self, config):
        super().__init__(config)
        self.patch_half_size = None
        self.extended_patch_half_size = None
        self.visualize = config['df_visualize']
        self.verbose = config['df_verbose']
        self.use_blur = not config['df_no_blur']
        self.blur_ksize = config['df_blur_ksize']
        self.source_amount = config['df_source_amount']

        self.set_patch_size(config['df_patch_size'])
        self.set_extended_patch_size(config['df_ext_patch_size'])

        self.image_depth = None
        self.image_color = None
        self.image_source_region = None
        self.image_source_region = None
        self.image_hole_region = None
        self.original_hole_region = None

        self.source_indices = None
        self.hole_boundary_indices = None

        self.i_max = None

        self.lowest_priority_index = None

        self.psi_p_hat = None
        self.psi_q_hat_d = None
        self.psi_q_hat_c = None

        self.reliability_d = None
        self.reliability_c = None

        self.psi_target = None

    def set_patch_size(self, patch_size):
        self.patch_half_size = int((patch_size - 1) / 2)

        if self.verbose:
            print("Filter size: {}".format(patch_size))

    def set_extended_patch_size(self, extended_patch_size):
        self.extended_patch_half_size = int((extended_patch_size - 1) / 2)

        if self.verbose:
            print("Extended Filter size: {}".format(extended_patch_size))

    def get_max_patch_half_size(self):
        return self.extended_patch_half_size

    def __call__(self, image_depth, image_color):
        self.initial_setup(image_color, image_depth)

        while self.hole_boundary_indices.shape[0] > 0:
            prev = np.nonzero(self.image_hole_region)[0].shape[0]

            self.compute_lowest_priority_index()

            self.most_similar_depth_index = self.get_most_similar_region_center_index(self.image_depth)
            self.most_similar_color_index = self.get_most_similar_region_center_index(self.image_color)

            self.define_psi_hats()

            self.compute_reliability()

            self.compute_psi_target()
            self.fill_target_into_depth_image()

            # update
            self.update_regions()
            self.update_hole_boundary()

            if self.visualize:
                self.visualize_process()

        # reverse initial mapping_network
        self.image_depth = np.where(self.image_depth > 0, self.i_max - self.image_depth + 1, 0)

        # apply blur
        if self.use_blur:
            blur = cv.GaussianBlur(self.image_depth, (self.blur_ksize, self.blur_ksize), 0)
            blur = np.reshape(blur, self.image_depth.shape)
            self.original_hole_region = np.reshape(self.original_hole_region, self.image_depth.shape)
            self.image_depth = np.where(self.original_hole_region > 0, blur, self.image_depth)

        return self.image_depth

    def initial_setup(self, image_color, image_depth):
        self.image_depth = image_depth
        self.image_color = image_color

        # correct shape if necessary
        if len(self.image_depth.shape) == 2:
            self.image_depth = np.reshape(self.image_depth, (self.image_depth.shape[0], self.image_depth.shape[1], 1))

        # map depth image from far = greatest values to near = greatest values. Hole values are 0
        self.i_max = np.max(self.image_depth)
        self.image_depth = np.where(self.image_depth > 0, self.i_max - self.image_depth + 1, 0)

        # get source and hole regions
        _, self.image_source_region = cv.threshold(self.image_depth, 0, 255, cv.THRESH_BINARY)
        _, self.image_hole_region = cv.threshold(self.image_depth, 0, 255, cv.THRESH_BINARY_INV)
        self.image_source_region = self.image_source_region.astype("uint8")
        self.image_hole_region = self.image_hole_region.astype("uint8")
        self.original_hole_region = np.copy(self.image_hole_region)

        # erode source region to gain better performance (more fields are filled per iteration)
        structuring_element = cv.getStructuringElement(
            cv.MORPH_RECT, (self.patch_half_size, self.patch_half_size))
        self.image_source_region = cv.erode(self.image_source_region, structuring_element)

        self.compute_source_indices()
        self.update_hole_boundary()

        if self.verbose:
            print("image size: {}".format(self.image_depth.shape[0:2]))
            print("Source region pixel amount: {}".format(self.source_indices.shape[0]))
            print("Hole   region pixel amount: {}".format(np.nonzero(
                self.image_hole_region[
                self.extended_patch_half_size:-self.extended_patch_half_size - 1,
                self.extended_patch_half_size:-self.extended_patch_half_size - 1
                ])[0].shape[0]))

    def compute_source_indices(self):
        """
        Computes the source indices used for most similar region searching.
        For performance reason, not all source region pixels can be used for this.

        Therefore this method determines
        """
        # get all source indices
        all_source_indices = np.nonzero(
            self.image_source_region[
            self.extended_patch_half_size:-self.extended_patch_half_size - 1,
            self.extended_patch_half_size:-self.extended_patch_half_size - 1
            ])
        total_amount_of_source_pixels = all_source_indices[0].shape[0]

        if self.verbose:
            print("total_amount_of_source_pixels: {}".format(total_amount_of_source_pixels))

        if total_amount_of_source_pixels <= self.source_amount:
            source_indices_region = all_source_indices
        else:
            source_indices_region = self.get_sliced_source_indices_region(all_source_indices,
                                                                          total_amount_of_source_pixels)

        # create indices array
        self.source_indices = np.full((source_indices_region[0].shape[0], 2), -1, dtype=int)

        # switched indexes to obtain convention
        self.source_indices[:, 0] = source_indices_region[1]
        self.source_indices[:, 1] = source_indices_region[0]

        # add extended patch size to map into original image domain
        self.source_indices += self.extended_patch_half_size

    def get_sliced_source_indices_region(self, all_source_indices, total_amount_of_source_pixels):
        # calc slice steps
        slice_step_quotient = total_amount_of_source_pixels / self.source_amount
        quotient_sqrt = math.sqrt(slice_step_quotient)

        slice_step_x = int(quotient_sqrt) if quotient_sqrt % 1 == 0 else int(quotient_sqrt) + 1
        slice_step_y = int(round(quotient_sqrt))

        # randomly switch x- and y-step
        if bool(random.getrandbits(1)):
            temp = slice_step_x
            slice_step_x = slice_step_y
            slice_step_y = temp

        # slice
        sliced_source_region = np.zeros(self.image_depth.shape[0:2])
        sliced_source_region[::slice_step_y, ::slice_step_x] = self.image_source_region[
                                                               ::slice_step_y, ::slice_step_x]

        # fill randomly up
        self.randomly_fill_up(sliced_source_region, all_source_indices, total_amount_of_source_pixels)

        if self.verbose:
            print("slice_step_x: {}, slice_step_y: {}".format(slice_step_x, slice_step_y))

        return np.nonzero(
            sliced_source_region[
            self.extended_patch_half_size:-self.extended_patch_half_size - 1,
            self.extended_patch_half_size:-self.extended_patch_half_size - 1
            ])

    def randomly_fill_up(self, sliced_source_region, all_source_indices, total_amount_of_source_pixels):
        sliced_pixel_amount = np.nonzero(sliced_source_region)[0].shape[0]

        amount_to_fill = max(self.source_amount - sliced_pixel_amount, 0)
        for i in range(amount_to_fill):
            random_index = random.randint(0, total_amount_of_source_pixels - 1)

            # Ensure index is not in sliced_source_region
            while True:  # do-while
                coord = np.array([all_source_indices[0][random_index], all_source_indices[1][random_index]])
                coord += self.extended_patch_half_size  # add extended patch size to map into original image domain
                if sliced_source_region[coord[0], coord[1]] == 0:
                    break
                random_index = (random_index + 1) % total_amount_of_source_pixels

            # set random source pixel in sliced region to 255
            sliced_source_region[coord[0], coord[1]] = 255

        if self.verbose:
            print("source pixel amount_to_fill: {}".format(amount_to_fill))

    def update_hole_boundary(self):
        contours, _ = cv.findContours(
            self.image_hole_region[self.extended_patch_half_size:-self.extended_patch_half_size - 1,
            self.extended_patch_half_size:-self.extended_patch_half_size - 1],
            cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        boundary_indices_amount = sum([array.shape[0] for array in contours])

        self.hole_boundary_indices = np.zeros((boundary_indices_amount, 2), dtype=int)
        index = 0
        for array in contours:
            self.hole_boundary_indices[index: index + array.shape[0]] = array[:, 0]
            index += array.shape[0]

        self.hole_boundary_indices += self.extended_patch_half_size

    def compute_lowest_priority_index(self):
        # lowest because out depth images are reversed, far is high value and near is small value
        lowest_priority = sys.maxsize
        lowest_priority_index = [-1, -1]

        for index in self.hole_boundary_indices:
            if self.image_hole_region[index[1], index[0]] == 0:
                raise RuntimeError("Value is 0! index: {}".format(index))

            # if is no real hole boundary value
            if np.sum(self.image_hole_region[index[1] - 1:index[1] + 2, index[0] - 1: index[0] + 2]) == 9 * 255:
                continue

            psi_p = self.image_depth[index[1] - self.patch_half_size:index[1] + self.patch_half_size + 1,
                    index[0] - self.patch_half_size:index[0] + self.patch_half_size + 1, 0]
            confidence_p = np.sum(psi_p) / psi_p.size
            i_avr = self.get_no_hole_average(psi_p)
            distance_p = (1 / 255) * (self.i_max - i_avr)
            priority = confidence_p * distance_p

            if priority < lowest_priority:
                lowest_priority = priority
                lowest_priority_index = index

        self.lowest_priority_index = np.array(lowest_priority_index)

        if self.verbose:
            print("lowest_priority_index: {}".format(lowest_priority_index))

    def get_most_similar_region_center_index(self, image):
        psi_region_center = image[
                            self.lowest_priority_index[1] - self.patch_half_size:
                            self.lowest_priority_index[1] + self.patch_half_size + 1,
                            self.lowest_priority_index[0] - self.patch_half_size:
                            self.lowest_priority_index[0] + self.patch_half_size + 1
                            ]

        lowest_ssd = sys.maxsize
        lowest_ssd_index = [-1, -1]

        for index_q in self.source_indices:
            if self.image_source_region[index_q[1], index_q[0]] == 0:
                raise RuntimeError("Value is 0! index: {}".format(index_q))

            psi_q = image[index_q[1] - self.patch_half_size:index_q[1] + self.patch_half_size + 1,
                    index_q[0] - self.patch_half_size:index_q[0] + self.patch_half_size + 1]
            ssd = np.sum((psi_region_center - psi_q) ** 2)

            if ssd < lowest_ssd:
                lowest_ssd = ssd
                lowest_ssd_index = index_q

        return np.array(lowest_ssd_index)

    def define_psi_hats(self):
        self.psi_p_hat = self.image_depth[
                         self.lowest_priority_index[1] - self.patch_half_size:
                         self.lowest_priority_index[1] + self.patch_half_size + 1,
                         self.lowest_priority_index[0] - self.patch_half_size:
                         self.lowest_priority_index[0] + self.patch_half_size + 1,
                         0
                         ]
        self.psi_q_hat_d = self.image_depth[
                           self.most_similar_depth_index[1] - self.patch_half_size:
                           self.most_similar_depth_index[1] + self.patch_half_size + 1,
                           self.most_similar_depth_index[0] - self.patch_half_size:
                           self.most_similar_depth_index[0] + self.patch_half_size + 1,
                           0
                           ]
        self.psi_q_hat_c = self.image_depth[
                           self.most_similar_color_index[1] - self.patch_half_size:
                           self.most_similar_color_index[1] + self.patch_half_size + 1,
                           self.most_similar_color_index[0] - self.patch_half_size:
                           self.most_similar_color_index[0] + self.patch_half_size + 1,
                           0
                           ]

        if self.verbose:
            print("psi_p_hat: \n{}".format(self.psi_p_hat.astype("uint16")))
            print("psi_q_hat_d: \n{}".format(self.psi_q_hat_d.astype("uint16")))
            print("psi_q_hat_c: \n{}".format(self.psi_q_hat_c.astype("uint16")))

    def compute_reliability(self):
        psi_p_hat_band = np.copy(self.image_depth[
                                 self.lowest_priority_index[1] - self.extended_patch_half_size:
                                 self.lowest_priority_index[1] + self.extended_patch_half_size + 1,
                                 self.lowest_priority_index[0] - self.extended_patch_half_size:
                                 self.lowest_priority_index[0] + self.extended_patch_half_size + 1,
                                 0
                                 ])
        psi_p_hat_band[self.patch_half_size:-self.patch_half_size,
        self.patch_half_size:-self.patch_half_size] = 0

        i_avr_p_hat = self.get_no_hole_average(self.psi_p_hat)
        i_avr_p_hat_band = self.get_no_hole_average(psi_p_hat_band, )
        i_avr_q_hat_d = self.get_no_hole_average(self.psi_q_hat_d)
        i_avr_q_hat_c = self.get_no_hole_average(self.psi_q_hat_c)

        error_in_d = abs(i_avr_q_hat_d - i_avr_p_hat)
        error_in_c = abs(i_avr_q_hat_c - i_avr_p_hat)
        error_band_d = abs(i_avr_q_hat_d - i_avr_p_hat_band)
        error_band_c = abs(i_avr_q_hat_c - i_avr_p_hat_band)

        self.reliability_d = 1 / (error_in_d * error_band_d) if error_in_d * error_band_d != 0 else 1
        self.reliability_c = 1 / (error_in_c * error_band_c) if error_in_c * error_band_c != 0 else 1

        if self.verbose:
            print("reliability d: {}, c: {}".format(self.reliability_d, self.reliability_c))

    def get_no_hole_average(self, psi):
        mask = np.where(psi > 0, 1, 0)
        psi_sum = np.sum(psi)
        no_hole_sum = np.sum(mask)

        if no_hole_sum > 0:
            return psi_sum / no_hole_sum
        else:
            return 0

    def compute_psi_target(self):
        weight_d = self.reliability_d / (self.reliability_d + self.reliability_c)
        weight_c = self.reliability_c / (self.reliability_d + self.reliability_c)
        ssd = math.sqrt(np.sum((self.psi_q_hat_d - self.psi_q_hat_c) ** 2)) / self.psi_q_hat_d.size

        # check if interpolate or not
        if ssd < 10:
            # calc mask (dont use hole region because of erode operation)
            mask = np.where(self.psi_q_hat_d == 0, 0, 1)
            mask = np.where(self.psi_q_hat_c == 0, 0, mask)

            self.psi_target = (self.psi_q_hat_d * weight_d + self.psi_q_hat_c * weight_c) * mask
            self.psi_target *= mask

        elif weight_d > weight_c:
            self.psi_target = self.psi_q_hat_d
        else:
            self.psi_target = self.psi_q_hat_c

        if self.verbose:
            print("weight d: {}, weight c: {}".format(weight_d, weight_c))
            print("psi_target: \n{}".format(self.psi_target.astype("uint16")))

    def fill_target_into_depth_image(self):
        # is reference, so its in self.cropped_depth
        self.psi_p_hat[:, :] = np.where(self.psi_p_hat > 0, self.psi_p_hat, self.psi_target)

        if self.verbose:
            print("image_depth: \n{}".format(self.psi_p_hat.astype("uint16")))

    def update_regions(self):
        # only update the hole region so that the psi_q_hat regions are always regions from original image
        _, psi_image_hole_region = cv.threshold(self.psi_p_hat, 0, 255, cv.THRESH_BINARY_INV)

        if self.verbose:
            print("hole region before: \n{}".format(
                self.image_hole_region[
                self.lowest_priority_index[1] - self.patch_half_size:
                self.lowest_priority_index[1] + self.patch_half_size + 1,
                self.lowest_priority_index[0] - self.patch_half_size:
                self.lowest_priority_index[0] + self.patch_half_size + 1
                ].astype("uint16")))
            print("hole region after : \n{}".format(psi_image_hole_region.astype("uint8")))

        self.image_hole_region[
        self.lowest_priority_index[1] - self.patch_half_size:
        self.lowest_priority_index[1] + self.patch_half_size + 1,
        self.lowest_priority_index[0] - self.patch_half_size:
        self.lowest_priority_index[0] + self.patch_half_size + 1
        ] = psi_image_hole_region

    def visualize_process(self):
        color = np.copy(self.image_color)
        cv.rectangle(color,
                     (self.lowest_priority_index[0] - self.patch_half_size,
                      self.lowest_priority_index[1] - self.patch_half_size),
                     (self.lowest_priority_index[0] + self.patch_half_size,
                      self.lowest_priority_index[1] + self.patch_half_size), (255, 0, 255), 2)
        cv.rectangle(color,
                     (self.most_similar_depth_index[0] - self.patch_half_size,
                      self.most_similar_depth_index[1] - self.patch_half_size),
                     (self.most_similar_depth_index[0] + self.patch_half_size,
                      self.most_similar_depth_index[1] + self.patch_half_size), (0, 255, 0), 2)
        cv.rectangle(color,
                     (self.most_similar_color_index[0] - self.patch_half_size,
                      self.most_similar_color_index[1] - self.patch_half_size),
                     (self.most_similar_color_index[0] + self.patch_half_size,
                      self.most_similar_color_index[1] + self.patch_half_size), (0, 0, 255), 2)
        color = color.astype("uint8")

        image_boundary = np.zeros(self.image_depth.shape[0:2])
        for index in self.hole_boundary_indices:
            image_boundary[index[1], index[0]] = 255

        image_source_indices = np.zeros(self.image_depth.shape[0:2])
        for index in self.source_indices:
            image_source_indices[index[1], index[0]] = 255

        labels = ["depth", "color", "boundary", "source indices"]
        images = [self.image_depth, color, image_boundary, image_source_indices]
        visualize_images(images, labels, ColorFormat.BGR, True)
