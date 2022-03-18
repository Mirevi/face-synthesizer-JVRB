from config import ConfigOptionPackage, ConfigOptionMetadata
from util.image_tools import ColorFormat


class NameCOP(ConfigOptionPackage):
    name_help = ''

    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'name', None, NameCOP.name_help, is_required=True),
        ]


class ContinueCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'continue_process', False, 'Continues the process instead of starting it from the beginning.'),
        ]


class OverwriteCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'overwrite', False, 'If generated Data should overwrite already existing data.'),
        ]


class VisualizeCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'visualize', False, 'If the process should be visualized.'),
        ]


class SaveDataCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'output_root', None, 'Path to directory where to save generated data.', is_required=True),
        ]


class LoadDataCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'input_root', None, 'Path to directory with input data.', is_required=True),
        ]


class DepthMaskCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(bool, 'no_depth_mask', False,
                                 'Do not use the depth image to mask out the background for calculations in model class.')
        ]


class ColorFormatCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(ColorFormat, 'color_format', ColorFormat.BGR,
                                 'Which color format to use for colored images.'),
        ]
