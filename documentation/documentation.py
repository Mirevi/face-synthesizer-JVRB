import numpy as np
import os
import sys
import ntpath
import time
import cv2 as cv

from config import ConfigOptionPackage, ConfigOptionMetadata, ConfigPackageProvider, NameCOP, SaveDataCOP, \
    ColorFormatCOP
from documentation.html import HTML
from util import util, tensor_to_image, map_image_values, save_image, ColorFormat, mkdirs
from subprocess import Popen, PIPE

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class DocumentationCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'no_write', False, 'do not write intermediate training results to txt file'),
            ConfigOptionMetadata(bool, 'no_print', False, 'do not print intermediate training results to config'),
            ConfigOptionMetadata(bool, 'no_html', False, 'do not save intermediate training results as html'),
            ConfigOptionMetadata(bool, 'no_visdom', False,
                                 'do not send intermediate training results to visdom server'),
        ]

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        if not options.no_write:
            metadata.extend([
                ConfigOptionMetadata(int, 'write_freq', 100, 'frequency of append training results to txt file'),
            ])
        if not options.no_print:
            metadata.extend([
                ConfigOptionMetadata(int, 'print_freq', 100, 'frequency of showing training results on console'),
            ])
        if not options.no_html:
            metadata.extend([
                ConfigOptionMetadata(int, 'html_freq', 400, 'frequency of saving training results to html'),
                ConfigOptionMetadata(int, 'html_image_width', 400, 'html display image size for visualizations'),
            ])
        if not options.no_visdom:
            metadata.extend([
                ConfigOptionMetadata(int, 'visdom_freq', 100, 'frequency of showing training results on screen'),
                ConfigOptionMetadata(str, 'visdom_url', 'http://localhost', 'visdom server url of the web display'),
                ConfigOptionMetadata(int, 'visdom_port', 8097, 'visdom port of the web display'),
                ConfigOptionMetadata(int, 'visdom_ncols', 5,
                                     'if positive, display all images in a single visdom web panel with certain number '
                                     'of images per row.'),
                ConfigOptionMetadata(str, 'visdom_env', 'main', 'visdom display environment name (default is "main")'),
            ])
        return metadata


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, color_format=ColorFormat.BGR):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
        color_format (ColorFormat)-- the format of color images

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = tensor_to_image(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio, color_format=color_format)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Documentation(ConfigPackageProvider):
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    @staticmethod
    def get_required_option_packages() -> list:
        return [DocumentationCOP, SaveDataCOP, NameCOP, ColorFormatCOP]

    # TODO refactor to add modular documentations (html, visdom, matplotlib)

    def __init__(self, config, continue_process):
        """Initialize the Visualizer class"""
        # validate config
        if self not in config:
            raise RuntimeError('Necessary Package Provider is not in current Config!')

        # get values from config
        self.name = config['name']
        output_root = config['output_root']
        self.save_root = os.path.join(output_root, self.name)
        self.color_format = config['color_format']
        self.use_write = not config['no_write']
        self.use_print = not config['no_print']
        self.use_html = not config['no_html']
        self.use_visdom = not config['no_visdom']
        self.evaluations = {"epochs": [], 'metric_names': [], 'metrics': []}

        if self.use_write:
            self.write_freq = config['write_freq']

            if not continue_process:
                writer_mode = 'w'
            else:
                writer_mode = 'a'

            # create a logging file to store training losses
            self.log_name_loss = os.path.join(self.save_root, 'loss_log.txt')
            with open(self.log_name_loss, writer_mode) as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

            # create a logging file to store evaluations
            self.log_name_evaluation = os.path.join(self.save_root, 'evaluation_log.txt')
            with open(self.log_name_evaluation, writer_mode) as log_file:
                now = time.strftime("%c")
                log_file.write('================ Model Evaluations (%s) ================\n' % now)

        if self.use_print:
            self.print_freq = config['print_freq']

        if self.use_html:
            self.html_freq = config['html_freq']
            self.html_image_width = config['html_image_width']
            self.html_location = os.path.join(self.save_root, 'web')
            self.html_img_location = os.path.join(self.html_location, 'images')

            print('create web directory %s...' % self.html_location)
            mkdirs([self.html_location, self.html_img_location])
            self.html_visual_labels = []

        if self.use_visdom:
            import visdom
            self.visdom_freq = config['visdom_freq']
            self.visdom_url = config['visdom_url']
            self.visdom_port = config['visdom_port']
            self.visdom_ncols = config['visdom_ncols']
            self.visdom_env = config['visdom_env']
            self.visdom_display_id = 0
            self.vis = visdom.Visdom(server=self.visdom_url, port=self.visdom_port, env=self.visdom_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start
        a new server at port < self.visdom_port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.visdom_port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def document_iteration(self, epoch: int, iteration: int, epoch_progress: float, visuals, losses):
        """documents the current iteration

        Parameters:
            epoch (int)            -- current epoch
            iteration (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            epoch_progress (float) -- progress (percentage) in the current epoch, between 0 to 1
            visuals (OrderedDict)  -- the visuals from a model
            losses (OrderedDict)   -- training losses from a model
        """
        self.document_losses(epoch, iteration, epoch_progress, losses)
        self.document_visuals(visuals, epoch, iteration)

    def document_losses(self, epoch, iteration, epoch_progress, losses):
        if self.use_write and iteration % self.write_freq == 0:
            self.write_current_losses(losses, epoch, iteration)

        if self.use_print and iteration % self.print_freq == 0:
            self.print_current_losses(losses, epoch, iteration)

        if self.use_visdom and iteration % self.visdom_freq == 0:
            self.document_losses_visdom(losses, epoch, epoch_progress)

    def write_current_losses(self, losses, epoch: int, iteration: int):
        """write current losses to losses.txt file

        Parameters:
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            epoch (int) -- current epoch
            iteration (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        """
        message = self.losses_to_string(losses, iteration, epoch)
        with open(self.log_name_loss, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_losses(self, losses, epoch: int, iteration: int):
        """print current losses on console

        Parameters:
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            epoch (int) -- current epoch
            iteration (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        """
        message = self.losses_to_string(losses, iteration, epoch)
        print(message)  # print the message

    @staticmethod
    def losses_to_string(losses, iteration, epoch: int):
        message = '(epoch: {}, iters: {}) '.format(epoch, iteration)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        return message

    def document_visuals(self, visuals, epoch: int, iteration: int = 0):
        if self.use_html and iteration % self.html_freq == 0:
            self.document_visuals_html(visuals, epoch)

        if self.use_visdom and iteration % self.visdom_freq == 0:
            self.document_visuals_visdom(visuals, epoch)

    def document_visuals_html(self, visuals, epoch: int):
        # save images to the disk
        for label, image in visuals.items():
            image_numpy = tensor_to_image(image[0])
            image_numpy = map_image_values(image_numpy, (0, 1), (0, 255))
            image_numpy = image_numpy.astype('uint8')

            if len(image_numpy.shape) >= 3 and image_numpy.shape[-1] > 4:
                raise RuntimeError("try to save numpy image with name '{}' and shape {}".format(
                    label, image_numpy.shape))
            img_path = os.path.join(self.html_img_location, 'epoch%.3d_%s.png' % (epoch, label))
            save_image(image_numpy, img_path, color_format=self.color_format)

        # update html
        self.html_visual_labels = [x for x, _ in visuals.items()]
        self.update_html(epoch)

    def document_visuals_visdom(self, visuals, epoch: int):
        if len(visuals) == 0:
            return

        if self.visdom_ncols <= 0:
            self.visdom_ncols = 1

        ncols = min(self.visdom_ncols, len(visuals))
        h, w = next(iter(visuals.values())).shape[:2]
        table_css = """<style>
                            table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; 
                            text-align: center}
                            table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                            </style>""" % (w, h)  # create a table css
        # create a table of images.
        title = self.name
        label_html = ''
        label_html_row = ''
        images = []
        idx = 0
        image_numpy = None
        for label, image in visuals.items():
            image_numpy = tensor_to_image(image[0])
            image_numpy = map_image_values(image_numpy, (0, 1), (0, 255))
            image_numpy = image_numpy.astype('uint8')
            label_html_row += '<td>%s</td>' % label
            if image_numpy.shape[2] == 1:
                image_numpy = cv.cvtColor(image_numpy, cv.COLOR_GRAY2RGB)
            elif image_numpy.shape[2] == 4:
                image_numpy = cv.cvtColor(image_numpy, cv.COLOR_RGBA2RGB)
            if self.color_format is ColorFormat.BGR:
                image_numpy = cv.cvtColor(image_numpy, cv.COLOR_BGR2RGB)
            images.append(image_numpy.transpose([2, 0, 1]))
            idx += 1
            if idx % ncols == 0:
                label_html += '<tr>%s</tr>' % label_html_row
                label_html_row = ''
        white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
        while idx % ncols != 0:
            images.append(white_image)
            label_html_row += '<td></td>'
            idx += 1
        if label_html_row != '':
            label_html += '<tr>%s</tr>' % label_html_row
        try:
            self.vis.images(images, nrow=ncols, win=str(self.visdom_display_id + 1), padding=2,
                            opts=dict(title='{} images epoch {}'.format(title, epoch)))
            label_html = '<table>%s</table>' % label_html
            self.vis.text(table_css + label_html, win=str(self.visdom_display_id + 2),
                          opts=dict(title='{} labels epoch {}'.format(title, epoch)))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def document_losses_visdom(self, losses, epoch: int, epoch_progress):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            losses (OrderedDict)   -- training losses stored in the format of (name, float) pairs
            epoch (int)            -- current epoch
            epoch_progress (float) -- progress (percentage) in the current epoch, between 0 to 1
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + epoch_progress)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            x = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
            y = np.array(self.plot_data['Y'])

            if x.shape[1] == 1:
                x = np.reshape(x, (x.shape[0]))
                y = np.reshape(y, (y.shape[0]))

            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=str(self.visdom_display_id))
        except VisdomExceptionBase:
            self.create_visdom_connections()

    def document_evaluation(self, epoch: int, evaluation):
        """document current evaluation;

        Parameters:
            epoch (int) -- current epoch
            evaluation (OrderedDict) -- training evaluation stored in the format of (name, dict) pairs
        """
        # register evaluation
        if len(self.evaluations['epochs']) == 0:
            self.evaluations['metric_names'] = [metric_name for metric_name in evaluation.keys()]
            self.evaluations['metrics'] = [{} for _ in evaluation.keys()]
            for metric_index, metric_name in enumerate(self.evaluations['metric_names']):
                self.evaluations['metrics'][metric_index] = \
                    {value_names: [] for value_names in evaluation[metric_name].keys()}

        self.evaluations['epochs'].append(epoch)
        for metric_index, metric_name in enumerate(self.evaluations['metric_names']):
            for value_name in self.evaluations['metrics'][metric_index].keys():
                self.evaluations['metrics'][metric_index][value_name].append(evaluation[metric_name][value_name])

        if self.use_write:
            self.write_evaluation(evaluation, epoch)

        if self.use_print:
            self.print_evaluation(evaluation, epoch)

        if self.use_html:
            self.update_html(epoch)

        if self.use_visdom:
            self.update_evaluations_visdom()

    def write_evaluation(self, evaluation, epoch: int):
        # save to file
        with open(self.log_name_evaluation, "a") as log_file:
            message = self.evaluation_to_string(evaluation, epoch)
            log_file.write('%s\n' % message)  # save the message
            log_file.write('%s\n' % "END")  # save the message

    def print_evaluation(self, evaluation, epoch: int):
        """print current evaluation on console;

        Parameters:
            evaluation (OrderedDict) -- training evaluation stored in the format of (name, dict) pairs
            epoch (int) -- current epoch
        """
        message = self.evaluation_to_string(evaluation, epoch)

        print('----- evaluation -----')
        print(message)  # print the message
        print('----------------------')

    def update_html(self, epoch):
        # update website
        webpage = HTML(self.html_location, 'Experiment name = %s' % self.name, refresh=60)
        for n in range(epoch, 0, -1):
            # header
            webpage.add_header('Epoch %d' % n)

            # images
            ims, txts, links = [], [], []
            for label in self.html_visual_labels:
                # image_numpy = tensor_to_image(image[0])
                img_path = 'epoch%.3d_%s.png' % (n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.html_image_width)

            # evaluations
            if n in self.evaluations['epochs']:
                x_labels = self.evaluations['metric_names']
                y_labels = []
                values = []

                for metric_index, metric_name in enumerate(self.evaluations['metric_names']):
                    for value_name in self.evaluations['metrics'][metric_index].keys():
                        if value_name not in y_labels:
                            y_labels.append(value_name)

                epoch_index = self.evaluations['epochs'].index(n)
                for value_name in y_labels:
                    row_values = []
                    for metric_index, metric_name in enumerate(self.evaluations['metric_names']):
                        if value_name in self.evaluations['metrics'][metric_index]:
                            row_values.append(
                                '{:.4f}'.format(self.evaluations['metrics'][metric_index][value_name][epoch_index])
                            )
                        else:
                            row_values.append('')
                    values.append(row_values)

                webpage.add_table(values, x_labels=x_labels, y_labels=y_labels)

        webpage.save()

    @staticmethod
    def evaluation_to_string(evaluation, epoch: int):
        message = 'epoch: {}\n'.format(epoch)
        for metric_name, metric_dict in evaluation.items():
            message += metric_name + '\n'
            for value_name, value in metric_dict.items():
                message += '\t{}: {:.4f}\n'.format(value_name, float(value))
        return message

    def update_evaluations_visdom(self):
        for metric_index, metric_name in enumerate(self.evaluations['metric_names']):
            data_legend = [value_name for value_name in self.evaluations['metrics'][metric_index].keys()]
            data_x = self.evaluations['epochs']
            data_y = [[self.evaluations['metrics'][metric_index][value_name][self.evaluations['epochs'].index(epoch)]
                       for value_name in data_legend] for epoch in data_x]

            data_x = np.stack([np.array(data_x)] * len(data_legend), 1)
            data_y = np.array(data_y)

            if data_x.shape[1] == 1:  # prevent visdom error that shapes are different although they are the same
                data_x = np.reshape(data_x, data_x.shape[0])
                data_y = np.reshape(data_y, data_y.shape[0])

            try:
                self.vis.line(
                    X=data_x,
                    Y=data_y,
                    opts={
                        'title': '{} {} metric'.format(self.name, metric_name),
                        'legend': data_legend,
                        'xlabel': 'epoch',
                        'ylabel': 'value'},
                    win=str(self.visdom_display_id + 3 + metric_index))
            except VisdomExceptionBase:
                self.create_visdom_connections()
