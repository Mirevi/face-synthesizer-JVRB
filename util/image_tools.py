# -*- coding: utf-8 -*-
"""
This Script contains different functionalities for image operations
"""
import numpy as np
import cv2 as cv
from enum import Enum

from numpy import uint8

from .util import print_numpy
import matplotlib.pyplot as plt

fig = plt.figure()


class ColorFormat(Enum):
    RGB = 'RGB'
    BGR = 'BGR'

    def __str__(self):
        return self.value


def tensor_to_image(tensor_image):
    image = tensor_image.cpu().clone().detach().numpy()
    if len(image.shape) == 2:
        shape = image.shape
        image = np.reshape(image, (1, shape[0], shape[1]))
    image = np.transpose(image, (1, 2, 0))
    image = image * 0.5 + 0.5

    return image


def map_image_values(image, old_range, new_range):
    """
    Maps the values of an image to a new range and datatype.

    Parameters:
        image (array)             -- input array
        old_range (tuple)         -- the value range of the current image
        new_range (tuple)         -- the desired new value range of the image
    """
    old_space_size = old_range[1] - old_range[0]
    new_space_size = new_range[1] - new_range[0]
    multiplier = new_space_size / old_space_size

    image -= old_range[0]
    image *= multiplier
    image += new_range[0]

    return image


def save_image(image_numpy, image_path, aspect_ratio=1.0, color_format=ColorFormat.BGR):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
        aspect_ratio (float)      -- the aspect ratio (default 1.0)
        color_format (ColorFormat)-- the format of color images
    """
    if aspect_ratio != 1.0:
        image_numpy = cv.resize(image_numpy, None, fx=aspect_ratio, fy=aspect_ratio, interpolation=cv.INTER_CUBIC)

    if color_format is ColorFormat.RGB and len(image_numpy.shape) == 3 and image_numpy.shape[2] != 1:
        image_numpy = switch_color_format(image_numpy)

    cv.imwrite(image_path, image_numpy)


def get_image_excerpt(image, bbox, padding=0, return_copy=True):
    excerpt_indices = {
        'y1': int(bbox['y']) - padding,
        'x1': int(bbox['x']) - padding,
        'y2': int(bbox['y'] + bbox['height']) + padding,
        'x2': int(bbox['x'] + bbox['width']) + padding,
    }

    # check boundaries
    if excerpt_indices['y1'] < 0 or excerpt_indices['x1'] < 0 or \
            excerpt_indices['y2'] >= image.shape[0] or \
            excerpt_indices['x2'] >= image.shape[1]:
        raise ValueError('Excerpt is not in image boundaries! '
                         'Excerpt indices are {}, image shape: {}'.format(excerpt_indices, image.shape))

    excerpt = image[
       excerpt_indices['y1']: excerpt_indices['y2'],
       excerpt_indices['x1']: excerpt_indices['x2']
    ]

    if return_copy:
        excerpt = np.copy(excerpt)

    return excerpt


def clip_16_bit_to_8_bit_uint(image, clipping_near, clipping_far):
    image = image.astype(float)
    image = map_image_values(image, (clipping_near, clipping_far), (0, 255))
    image = np.where(image < 0, 0, image)
    image = np.where(image > 255, 0, image)
    image = image.astype(uint8)
    return image


def switch_color_format(image):
    """
    Switches the color format of the given image between RGB <-> BGR or RGBD <-> BGRD.

    Parameters:
        image (numpy array)-- An image
    """
    if len(image.shape) != 3 or image.shape[2] == 1:
        raise RuntimeError("Unsupported channel size 1 for preprocessing of color format.")

    if image.dtype == float:
        image = image.astype('float32')

    if image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    elif image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
    else:
        raise RuntimeError("Unsupported channel size {} for preprocessing of color format.".format(image.shape[2]))

    return image


def convert_into_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] > 1 and image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] > 1 and image.shape[2] == 4:
        image = cv.cvtColor(image, cv.COLOR_BGRA2GRAY)

    return image


def visualize_images(images: list, labels: list = None, color_format=ColorFormat.RGB, continuous_drawing=True,
                     pause_interval=0.001):
    """
    Visualizes all given images.
    :param images: A list of images.
    :param labels: A list of labels.
    :param color_format: The color format of colored images in the images list.
    :param continuous_drawing: True if the visualize function is called in iterations.
            False if the Visualization is called once or should wait for user prompt.
    :param pause_interval: Seconds to wait until continuing the program when continuous_drawing is True.
    """
    global fig

    fig.clf()

    axis_offset = 0
    for i in range(len(images)):
        current_image = images[i]

        # convert image color format if necessary
        if color_format == ColorFormat.BGR and len(current_image.shape) == 3 and current_image.shape[2] >= 3:
            current_image = switch_color_format(current_image)

        # get label if exists
        current_label = ""
        if labels is not None and len(labels) > i:
            current_label = labels[i]

        # add data to plot
        if len(current_image.shape) == 3 and current_image.shape[2] == 4:
            add_image_to_figure(fig, current_image[:, :, 0:3], i + axis_offset, current_label + " - Color")
            axis_offset += 1
            add_image_to_figure(fig, current_image[:, :, 3], i + axis_offset, current_label + " - Depth")
        else:
            add_image_to_figure(fig, current_image, i + axis_offset, current_label)

    if continuous_drawing:
        plt.draw()
        plt.pause(pause_interval)
    else:
        plt.show()


def add_image_to_figure(fig, current_image, index, current_label):
    current_axis = fig.add_subplot(221 + index)
    current_axis.set_title(current_label)
    current_axis.imshow(current_image)


def visualize_images_cv(images: list, labels: list = None, color_format=ColorFormat.RGB,
                        continuous_drawing=True):
    if labels is None:
        labels = [str(i) for i in range(len(images))]

    for i in range(len(images)):
        current_image = images[i]

        # convert image color format if necessary
        if color_format == ColorFormat.RGB and len(current_image.shape) == 3 and current_image.shape[2] >= 3:
            current_image = switch_color_format(current_image)

        # show image
        cv.imshow(labels[i], current_image)

    if continuous_drawing:
        cv.waitKey(1)
    else:
        cv.waitKey(0)


def visualize_tensor_images(tensor_images: list, labels: list = None, color_format=ColorFormat.RGB,
                            continuous_drawing=True):
    """
    Visualizes all given Tensor images.
    :param tensor_images: A list of Tensor images
    :param labels: A list of labels
    :param color_format: The color format of colored images in the images list.
    :param continuous_drawing: True if the visualize function is called in iterations.
            False if the Visualization is called once or should wait for user prompt.
    """
    images = []

    for i in range(len(tensor_images)):
        current = tensor_to_image(tensor_images[i])
        images.append(current)

    visualize_images(images, labels, color_format, continuous_drawing)


def visualize_tensor_images_cv(tensor_images: list, labels: list = None, color_format=ColorFormat.RGB,
                               continuous_drawing=True):
    """
    Visualizes all given Tensor images with cv.
    :param tensor_images: A list of Tensor images
    :param labels: A list of labels
    :param color_format: The color format of colored images in the images list.
    :param continuous_drawing: True if the visualize function is called in iterations.
            False if the Visualization is called once or should wait for user prompt.
    """
    images = []

    for i in range(len(tensor_images)):
        current = tensor_to_image(tensor_images[i])
        images.append(current)

    visualize_images_cv(images, labels, color_format, continuous_drawing)
