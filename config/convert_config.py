import os

from config.base_config import StandardConfig, ArgparseConfig, COPWithModifiableDefaults, ConfigDefaultModification, \
    ConfigOptionMetadata, ConfigPackageProvider
from config.config_option_packages import OverwriteCOP, VisualizeCOP, SaveDataCOP, NameCOP, ContinueCOP, LoadDataCOP
from data.preprocessing.depthFilling import DepthFillerCOP, depth_filler_class, DepthFillingCOP
from tracking.eye_tracking import EyeTrackingMethodCOP, eye_tracker_class
from tracking.face_aabbox_tracking.face_aabbox_tracking import FaceAABBoxTrackingMethodCOP, face_aabbox_tracker_class, \
    FaceAABBoxTrackingMethod
from tracking.face_tracking import FaceTrackingMethodCop, face_tracker_class


class ConvertCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'output_image_size', 512, 'The image size of the output images.'),
            ConfigOptionMetadata(int, 'padding', 20,
                                 'Padding which is additionally added to the crop region. Does not affect output image size!'),
            ConfigOptionMetadata(int, 'feature_image_line_thickness', 4,
                                 'Determines the thickness of the lines of the feature image.'),
            ConfigOptionMetadata(float, 'horizontal_fov', 90.0, 'The horizontal field-of-view of the camera used to '
                                                                'capture the images.'),
            ConfigOptionMetadata(int, 'depth_padding', 20, 'The depth padding value (mm) for the clipping near when '
                                                           'converting the 16 bit depth to 8 bit depth.'),
            ConfigOptionMetadata(int, 'face_depth', 200, 'The depth of the face (mm) to determine the clipping far'
                                                         'when converting the 16 bit depth to 8 bit depth.'),
            ConfigOptionMetadata(float, 'pca_image_components', 0.99999,
                                 'Number of components to keep in images. If 0 < pca_image_components < 1, '
                                 'select the number of components such that the amount of variance that needs to be '
                                 'explained is greater than the percentage specified by pca_image_components.'),
            ConfigOptionMetadata(float, 'pca_landmarks_components', 0.99999,
                                 'Number of components to keep in landmarks. If 0 < pca_landmarks_components < 1, '
                                 'select the number of components such that the amount of variance that needs to be '
                                 'explained is greater than the percentage specified by pca_landmarks_components.'),
        ]

    @staticmethod
    def get_default_modifications() -> list:
        input_root = os.path.join(os.path.abspath(os.curdir), 'datasets', 'unprocessed')
        output_root = os.path.join(os.path.abspath(os.curdir), 'datasets', 'processed')
        return [
            ConfigDefaultModification(LoadDataCOP, 'input_root', input_root),
            ConfigDefaultModification(SaveDataCOP, 'output_root', output_root),
        ]


class ConvertConfigProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        NameCOP.name_help = 'The name of the data dir to process.'
        return [ConvertCOP, NameCOP, OverwriteCOP, VisualizeCOP, ContinueCOP, SaveDataCOP, LoadDataCOP,
                EyeTrackingMethodCOP, FaceTrackingMethodCop, DepthFillingCOP]

    @staticmethod
    def get_conditional_providers(options) -> list:
        providers = [
            eye_tracker_class(options.eye_tracking_method),
            face_tracker_class(options.face_tracking_method),
            face_aabbox_tracker_class(FaceAABBoxTrackingMethod.SynergyNet),
            depth_filler_class(options.df_algorithm),
        ]
        return providers


class ConvertStandardConfig(StandardConfig):
    def __init__(self):
        super().__init__()
        self.add_package_provider(ConvertConfigProvider)


class ConvertArgparseConfig(ArgparseConfig):
    def __init__(self, parser_args=None):
        super().__init__(parser_args)
        self.add_package_provider(ConvertConfigProvider)


if __name__ == '__main__':
    from tracking.eye_tracking import EyeTrackingMethod

    config = ConvertStandardConfig()
    config['eye_tracking_method'] = EyeTrackingMethod.Infrared
    config["name"] = 'train'
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))

    args = ['--name', 'train']
    config = ConvertArgparseConfig(args)
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))
