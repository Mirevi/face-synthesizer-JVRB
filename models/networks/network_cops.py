from config import ConfigOptionPackage, ConfigOptionMetadata


class NetworkCOP(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(int, 'input_nc', 1, '# of input image channels: 3 for RGB and 1 for grayscale'),
            ConfigOptionMetadata(int, 'output_nc', 4, '# of output image channels: 3 for RGB and 1 for grayscale'),
        ]
