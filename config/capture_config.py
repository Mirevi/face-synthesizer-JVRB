import os

from config import StandardConfig, ArgparseConfig, COPWithModifiableDefaults, ConfigOptionMetadata, \
    ConfigDefaultModification, ConfigPackageProvider
from config.config_option_packages import NameCOP, SaveDataCOP, OverwriteCOP, VisualizeCOP, ContinueCOP
from tracking.face_aabbox_tracking import FaceAABBoxTrackingMethod, face_aabbox_tracker_class
from tracking.face_alignment_tracking import FaceAlignmentTrackingMethod, face_alignment_tracker_class


class CaptureCOP(COPWithModifiableDefaults):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'train_image_amount', 1500, 'The Amount of images for training to capture.'),
            ConfigOptionMetadata(int, 'eval_image_amount', 250, 'The Amount of images for evaluation to capture.'),
            ConfigOptionMetadata(int, 'image_id_offset', 0,
                                 'An id offset. The naming of the captured files will start at this value + 1.'),
            ConfigOptionMetadata(FaceAlignmentTrackingMethod, 'face_align_tracking_method',
                                 FaceAlignmentTrackingMethod.SynergyNet,
                                 'The Face Alignment Tracking method to use to sort out unsuitable data.'),
            ConfigOptionMetadata(float, 'max_yaw', 10.0, 'The maximum yaw angle for suitable data.'),
            ConfigOptionMetadata(float, 'max_pitch', 10.0, 'The maximum pitch angle for suitable data.'),
            ConfigOptionMetadata(float, 'max_roll', 10.0, 'The maximum roll angle for suitable data.'),

            # only allow tracker which do not need landmark data (would be slow)
            ConfigOptionMetadata(FaceAABBoxTrackingMethod, 'face_aabbox_tracking_method',
                                 FaceAABBoxTrackingMethod.SynergyNet,
                                 'The face axis aligned bounding box tracking method to use.',
                                 choices=[FaceAABBoxTrackingMethod.SynergyNet]),
        ]

    @staticmethod
    def get_default_modifications() -> list:
        return [
            ConfigDefaultModification(SaveDataCOP, 'output_root',
                                      os.path.join(os.path.abspath(os.curdir), 'datasets', 'unprocessed')),
        ]


class CaptureConfigProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        NameCOP.name_help = 'The name of the save directory.'
        return [CaptureCOP, NameCOP, OverwriteCOP, VisualizeCOP, ContinueCOP, SaveDataCOP]

    @staticmethod
    def get_conditional_providers(options) -> list:
        providers = []
        providers.extend([face_aabbox_tracker_class(options.face_aabbox_tracking_method)])
        providers.extend([face_alignment_tracker_class(options.face_align_tracking_method)])
        return providers


class CaptureStandardConfig(StandardConfig):
    def __init__(self):
        super().__init__()
        self.add_package_provider(CaptureConfigProvider)


class CaptureArgparseConfig(ArgparseConfig):
    def __init__(self, parser_args=None):
        super().__init__(parser_args)
        self.add_package_provider(CaptureConfigProvider)


if __name__ == '__main__':
    config = CaptureStandardConfig()
    config["name"] = 'new_data'
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))

    args = ['--name', 'new_data']
    config = CaptureArgparseConfig(args)
    config.gather_options()
    config.print(detailed_package_description=False)
    print(len(vars(config.options)))
