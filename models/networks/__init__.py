from config import ConfigPackageProvider
from .generator import *
from .discriminator import *
from .mapping_network import *
from .network_cops import NetworkCOP
from .gan_loss import GANLossCOP, GANMode, GANLoss
from .initialization import NetworkInitializationCOP, InitializationType, init_weights_with_config_values, init_weights
from .normalization_layer import NormalizationLayerCOP, NormalizationType, get_norm_layer_from_config, get_norm_layer
from .scheduler import SchedulerCOP, LRPolicy, get_scheduler
from .traceable_network import TraceableNetwork


class NetworkBasePackageProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        return [NormalizationLayerCOP, NetworkInitializationCOP, SchedulerCOP, NetworkCOP]


class ConvGANPackageProvider(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        return [GeneratorCOP, DiscriminatorCOP, GANLossCOP]

    @staticmethod
    def get_required_providers() -> list:
        return [NetworkBasePackageProvider]
