import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys


class ConfigOptionMetadata:
    def __init__(self, dtype, name: str, default, help: str, is_required: bool = False, choices=None,
                 is_constant: bool = False):
        self.dtype = dtype
        self.name = name
        self.default = default
        self.help = help
        self.is_required = is_required
        self.choices = choices
        self.is_constant = is_constant
        self.value_was_explicitly_changed = False

        # validate
        if self.default is not None and type(self.default) is not self.dtype:
            raise TypeError("default '{}' is not of dtype '{}'".format(default, dtype))

        # validate all choices are from dtype
        if self.choices is not None and not all(isinstance(i, self.dtype) for i in self.choices):
            raise TypeError("A choice from choices is not of dtype '{}'".format(self.dtype))

        # validate default in choice
        if self.default is not None and self.choices is not None and self.default not in self.choices:
            raise ValueError("default '{}' is not one of choices '{}'".format(self.default, self.choices))


class ConfigOptionPackage(ABC):
    @staticmethod
    @abstractmethod
    def get_options_metadata() -> list:
        return []

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        metadata = []
        return metadata


class ConfigDefaultModification:
    def __init__(self, package: ConfigOptionPackage.__class__, name: str, default):
        self.package = package
        self.name = name
        self.default = default

        # validate package class
        if self.package is not None and not issubclass(self.package, ConfigOptionPackage):
            raise TypeError("given package '{}' is not a subclass of '{}'".format(self.package, ConfigOptionPackage))


class COPWithModifiableDefaults(ConfigOptionPackage, ABC):
    @staticmethod
    def get_default_modifications() -> list:
        return []

    @staticmethod
    def get_conditional_default_modifications(options) -> list:
        modifications = []
        return modifications


class ConfigPackageProvider(ABC):
    @staticmethod
    def get_required_option_packages() -> list:
        return []

    @staticmethod
    def get_conditional_option_packages(options) -> list:
        packages = []
        return packages

    @staticmethod
    def get_required_providers() -> list:
        return []

    @staticmethod
    def get_conditional_providers(options) -> list:
        providers = []
        return providers


class BaseConfig(ABC):
    def __init__(self):
        self.options = type('', (), {})()
        self.options_metadata = {}
        self.package_provider = []
        self.option_packages = []

    def add_package_providers(self, providers: list):
        # validate class
        for provider in providers:
            if not issubclass(provider, ConfigPackageProvider):
                raise TypeError("Class '{}' is not a subclass of '{}'".format(provider, ConfigPackageProvider))

        # add all providers
        for provider in providers:
            self.add_package_provider(provider)

    def add_package_provider(self, provider: ConfigPackageProvider.__class__):
        # validate class
        if not issubclass(provider, ConfigPackageProvider):
            raise TypeError("Class '{}' is not a subclass of '{}'".format(provider, ConfigPackageProvider))

        # return if package is already registered
        if provider in self.package_provider:
            return

        # get all necessary provider
        necessary_provider = self.get_necessary_provider(provider, False)

        # add necessary provider
        for current in necessary_provider:
            self.add_option_packages(current.get_required_option_packages())
            self.package_provider.append(current)

    def get_necessary_provider(self, provider: ConfigPackageProvider.__class__, get_conditional):
        necessary_provider = [provider]
        index = 0
        while index < len(necessary_provider):
            current = necessary_provider[index]
            index += 1

            current_necessaries = current.get_conditional_providers(
                self.options) if get_conditional else current.get_required_providers()

            # validate necessary_provider
            for provider in current_necessaries:
                if not issubclass(provider, ConfigPackageProvider):
                    raise TypeError("Class {} is not a subclass of {}.".format(provider, ConfigPackageProvider))

            # add current_necessaries unregistered provider
            for provider in current_necessaries:
                if provider not in self.package_provider and provider not in necessary_provider:
                    necessary_provider.append(provider)

        return necessary_provider

    def add_option_packages(self, packages: list):
        # validate class
        for package in packages:
            if not issubclass(package, ConfigOptionPackage):
                raise TypeError("Class '{}' is not a subclass of '{}'".format(package, ConfigOptionPackage))

        # add all packages
        for package in packages:
            self.add_option_package(package)

    def add_option_package(self, package: ConfigOptionPackage.__class__):
        # validate class
        if not issubclass(package, ConfigOptionPackage):
            raise TypeError("Class '{}' is not a subclass of '{}'".format(package, ConfigOptionPackage))

        # return if package is already registered
        if package in self.option_packages:
            return

        # get all necessary packages
        necessary_packages = self.get_necessary_packages(package)

        # add packages and options
        for package in necessary_packages:
            self.add_multiple_option_metadata(package.get_options_metadata())
            self.option_packages.append(package)

        # get default modifications
        for package in necessary_packages:
            # continue if not COPWithModifiableDefaults
            if not issubclass(package, COPWithModifiableDefaults):
                continue

            self.apply_default_modifications(package.get_default_modifications())

    def get_necessary_packages(self, package: ConfigOptionPackage.__class__):
        packages = [package]
        index = 0
        while index < len(packages):
            current = packages[index]
            index += 1

            # continue if not a COPWithModifiableDefaults
            if not issubclass(current, COPWithModifiableDefaults):
                continue

            # get modifications
            default_modifications = current.get_default_modifications()

            # add packages of modifications if not current and already in option_packages list
            for modification in default_modifications:
                # validate modification
                if not isinstance(modification, ConfigDefaultModification):
                    raise TypeError("object '{}' is not of type '{}'".format(modification, ConfigDefaultModification))

                # continue if modification.package is None
                if modification.package is None:
                    continue

                # validate modification_package
                if not issubclass(modification.package, ConfigOptionPackage):
                    raise TypeError(
                        "Class '{}' is not a subclass of '{}'".format(modification.package, ConfigOptionPackage))

                # add modification.package to packages if not current, not already in option_packages
                # and not already in packages
                if modification.package is not current \
                        and modification.package not in self.option_packages \
                        and modification.package not in packages:
                    packages.append(modification.package)

        return packages

    def add_multiple_option_metadata(self, metadata: list):
        # validate class
        for current in metadata:
            if not isinstance(current, ConfigOptionMetadata):
                raise TypeError("object '{}' is not of type '{}'".format(current, ConfigOptionMetadata))

        for current in metadata:
            self.add_option_metadata(current)

    def add_option_metadata(self, metadata: ConfigOptionMetadata):
        # validate class
        if not isinstance(metadata, ConfigOptionMetadata):
            raise TypeError("object '{}' is not of type '{}'".format(metadata, ConfigOptionMetadata))

        # validate all choices are from dtype
        if metadata.choices is not None and not all(isinstance(i, metadata.dtype) for i in metadata.choices):
            raise TypeError("A choice from choices of ConfigOptionMetadata with name '{}' is not of dtype '{}'".format(
                metadata.name, metadata.dtype
            ))

        # validate consistency
        if metadata.default is not None:
            if not issubclass(type(metadata.default), metadata.dtype):
                raise TypeError("default '{}' of ConfigOptionMetadata with name '{}' is not of dtype '{}'".format(
                    metadata.default, metadata.name, metadata.dtype
                ))

            if metadata.choices is not None and metadata.default not in metadata.choices:
                raise ValueError(
                    "default '{}' of ConfigOptionMetadata with name '{}' is not one of choices '{}'".format(
                        metadata.default, metadata.name, metadata.choices
                    ))

        # add metadata
        self.options_metadata[metadata.name] = metadata

        # add option
        self.create_and_add_option(metadata)

    @abstractmethod
    def create_and_add_option(self, metadata: ConfigOptionMetadata):
        pass

    def apply_default_modifications(self, modifications: list):
        # validate class
        for modification in modifications:
            if not isinstance(modification, ConfigDefaultModification):
                raise TypeError("object '{}' is not of type '{}'".format(modification, ConfigDefaultModification))

        # apply modifications
        for modification in modifications:
            self.apply_default_modification(modification)

    def apply_default_modification(self, modification: ConfigDefaultModification):
        # validate class
        if not isinstance(modification, ConfigDefaultModification):
            raise TypeError("object '{}' is not of type '{}'".format(modification, ConfigDefaultModification))

        # check for package registration
        if modification.package is not None and modification.package not in self.option_packages:
            raise ValueError("package '{}' of modification was not added to this config.".format(
                modification.package))

        # validate consistency
        if modification.package is not None:
            package_options = modification.package.get_options_metadata()
            package_conditional_options = modification.package.get_conditional_options_metadata(self.options)

            # try to find option in package
            found_option = False
            for option in package_options:
                found_option = found_option or option.name == modification.name
            for option in package_conditional_options:
                found_option = found_option or option.name == modification.name

            # if not found raise error
            if not found_option:
                raise ValueError(
                    "cannot find option with name '{}' in ConfigOptionPackage '{}' in current config state.".format(
                        modification.name, modification.package
                    ))

        # validate existence
        if modification.name not in self.options_metadata:
            raise ValueError("cannot find option with name '{}' in config.".format(modification.name))

        # validate dtype
        if not type(modification.default) is self.options_metadata[modification.name].dtype:
            raise TypeError("modifications (name: '{}') default '{}' is not of metadata dtype '{}'".format(
                modification.name, modification.default, self.options_metadata[modification.name].dtype
            ))

        # set and apply default value
        option_metadata = self.options_metadata[modification.name]
        option_metadata.default = modification.default
        if not option_metadata.value_was_explicitly_changed:
            self.set_option_value(modification.name, modification.default)
            option_metadata.value_was_explicitly_changed = False

    def gather_options(self):
        # check required options
        self.validate_required_options()

        # add necessary conditional provider
        index = 0
        while index < len(self.package_provider):
            provider = self.package_provider[index]
            index += 1
            necessary_provider = self.get_necessary_provider(provider, True)[1:]
            self.add_package_providers(necessary_provider)

        # add conditional option packages from provider
        for provider in self.package_provider:
            self.add_option_packages(provider.get_conditional_option_packages(self.options))

        # add conditional options from packages
        for package in self.option_packages:
            metadata = package.get_conditional_options_metadata(self.options)
            self.add_multiple_option_metadata(metadata)

        # apply conditional default modifications
        for package in self.option_packages:
            # continue if not COPWithModifiableDefaults
            if not issubclass(package, COPWithModifiableDefaults):
                continue

            self.apply_default_modifications(package.get_conditional_default_modifications(self.options))

        # check required options
        self.validate_required_options()

    def validate_required_options(self):
        for metadata in self.options_metadata.values():
            if metadata.is_required and self[metadata.name] is None:
                raise RuntimeError("required option with name '{}' is None".format(metadata.name))

    def __getitem__(self, item: str):
        if not hasattr(self.options, item):
            raise ValueError("An Attribute with name '{}' doesnt exist.".format(item))

        return getattr(self.options, item)

    def __setitem__(self, key, value):
        # validate existence
        if not hasattr(self.options, key):
            raise ValueError("An Attribute with name '{}' doesnt exist.".format(key))

        # validate no constant
        if self.options_metadata[key].is_constant:
            raise AttributeError("The option with name '{}' is a constant value and cannot be changed!".format(key))

        # validate dtype
        if type(value) is not self.options_metadata[key].dtype:
            raise TypeError("The value '{}' is not of type '{}'".format(
                value, self.options_metadata[key].dtype))

        # validate value in choices
        if self.options_metadata[key].choices is not None and value not in self.options_metadata[key].choices:
            raise ValueError("The value '{}' is not a valid choice from choices '{}'".format(
                value, self.options_metadata[key].choices))

        self.options_metadata[key].value_was_explicitly_changed = True
        self.set_option_value(key, value)

    def get_options_from_provider(self, all_providers: ConfigPackageProvider.__class__):
        if all_providers not in self:
            return {}

        options_dict = {}

        # get all required and conditional providers
        index = 0
        all_providers = [all_providers]
        while index < len(all_providers):
            current = all_providers[index]
            index += 1
            necessary_providers = current.get_required_providers()
            necessary_providers.extend(current.get_conditional_providers(self.options))

            for provider in necessary_providers:
                if provider not in all_providers:
                    all_providers.append(provider)

        # get all required and conditional packages from all_providers
        all_packages = []
        for provider in all_providers:
            necessary_packages = provider.get_required_option_packages()
            necessary_packages.extend(provider.get_conditional_option_packages(self.options))

            for package in necessary_packages:
                if package not in all_packages:
                    all_packages.append(package)

        # add options from package to options_dict
        for package in all_packages:
            options_dict = {**options_dict, **self.get_options_from_package(package)}

        return options_dict

    def get_options_from_package(self, package: ConfigOptionPackage.__class__):
        if package not in self:
            return {}

        options_dict = {}
        package_metadata = package.get_options_metadata()
        package_metadata.extend(package.get_conditional_options_metadata(self.options))

        for metadata in package_metadata:
            options_dict[metadata.name] = self[metadata.name]

        return options_dict

    @abstractmethod
    def set_option_value(self, name, value):
        pass

    def __str__(self, without_constants=False):
        message = ''
        for k, v in sorted(vars(self.options).items()):
            if without_constants and self.options_metadata[k].is_constant:
                continue

            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        return message[:-1]

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.options_metadata
        elif isinstance(item, ConfigOptionPackage):
            return item.__class__ in self.option_packages
        elif isinstance(item, ConfigPackageProvider):
            return item.__class__ in self.package_provider
        elif issubclass(item, ConfigOptionPackage):
            return item in self.option_packages
        elif issubclass(item, ConfigPackageProvider):
            return item in self.package_provider

        return False

    def print(self, detailed_package_description=False):
        print('---------------- Config -----------------')
        print('Config Option Packages:')
        packages = sorted(self.option_packages, key=lambda package: package.__name__)
        if detailed_package_description:
            for package in packages:
                print('\t{}'.format(str(package)))
        else:
            package_names = [package.__name__ for package in packages]
            line = '\t'
            for i, name in enumerate(package_names):
                line += '{}, '.format(name)

                if len(line) > 120 or i == len(package_names) - 1:
                    print(line[:-2])
                    line = '\t'
        print('-----------------------------------------')
        print(self.__str__())
        print('----------------- End -------------------')

    def save_to_disk(self, directory_path, filename='Config.txt'):
        message = self.__str__(True)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'wt') as opt_file:
            opt_file.write(message)


class StandardConfig(BaseConfig):
    @staticmethod
    def string_to_dict(s):
        arg_dict = {}
        lines = s.split('\n')
        for line in lines:
            arguments = line.split(': ')
            if len(arguments) == 1:
                arguments += ['']
            attribute_name, value = arguments

            # remove unnecessary whitespaces
            attribute_name = attribute_name.strip()
            value = value.strip()
            # set value in dict
            arg_dict[attribute_name] = value

        return arg_dict

    def __init__(self, args: dict = None):
        super().__init__()
        # set args if none
        if args is None:
            args = {}

        # validate args
        if not isinstance(args, dict):
            raise TypeError('The given args value is not a dictionary!')

        self.args = args

    def create_and_add_option(self, metadata: ConfigOptionMetadata):
        key = metadata.name
        value = metadata.default

        # set value and explicit flag if a value is given in args
        if key in self.args:
            value = self.args[key]
            if type(value) is not metadata.dtype:
                try:
                    value = self.string_to_dtype(value, metadata.dtype)
                except:
                    pass
            metadata.value_was_explicitly_changed = True

        # validate value dtype
        if value is not None and type(value) is not metadata.dtype:
            raise TypeError(
                "The given value '{}' of option '{}' is not of dtype '{}'".format(value, key, metadata.dtype))

        # validate value in choices
        if metadata.choices is not None and value not in metadata.choices:
            raise ValueError("The given value '{}' of option '{}' is not a valid choice from '{}'".format(
                value, key, metadata.choices))

        # set default value if is_constant
        if metadata.is_constant:
            value = metadata.default

        self.set_option_value(key, value)

    def set_option_value(self, name, value):
        setattr(self.options, name, value)

    def set_values_from_string(self, s: str, ignore_unexpected_attributes=False):
        lines = s.split('\n')

        # check if all attributes exists
        if not ignore_unexpected_attributes:
            self.validate_attributes_in_lines(lines)

        # set all values
        for line in lines:
            attribute_name, value = self.attribute_name_and_value_from_line(line)
            if hasattr(self.options, attribute_name):
                self[attribute_name] = value

    def validate_attributes_in_lines(self, lines):
        for line in lines:
            attribute_name, _ = self.attribute_name_and_value_from_line(line)
            if not hasattr(self.options, attribute_name):
                raise ValueError("An Attribute with name '{}' doesnt exist.".format(attribute_name))

    def attribute_name_and_value_from_line(self, line):
        # split line into attribute_name and value
        arguments = line.split(': ')
        if len(arguments) == 1:
            arguments += ['']
        attribute_name, value = arguments

        # remove unnecessary whitespaces
        attribute_name = attribute_name.strip()
        value = value.strip()

        # to matching primitive datatype
        if attribute_name in self.options_metadata:
            value = self.string_to_dtype(value, self.options_metadata[attribute_name].dtype)

        return attribute_name, value

    @staticmethod
    def string_to_dtype(s:str, dtype):
        if dtype is bool:
            if s.lower() == 'true':
                return True
            elif s.lower() == 'false':
                return False
            else:
                raise ValueError("Cannot convert string '{}' into boolean!".format(s))

        return dtype(s)


class ArgparseConfig(BaseConfig):
    def __init__(self, parser_args=None):
        super().__init__()
        self.parser_args = sys.argv[1:] if parser_args is None else parser_args
        self.parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    def create_and_add_option(self, metadata: ConfigOptionMetadata):
        argparse_key = '--{}'.format(metadata.name)

        if metadata.dtype == bool and not metadata.is_required and not metadata.default:
            self.parser.add_argument(
                argparse_key,
                action='store_true',
                help=metadata.help,
            )
        else:
            self.parser.add_argument(
                argparse_key,
                type=metadata.dtype,
                default=metadata.default,
                help=metadata.help
            )

        # set metadata.value_was_explicitly_changed if a value is given
        if argparse_key in self.parser_args:
            metadata.value_was_explicitly_changed = True

        # add default to parser_args if is_required, default exists and if no explicit value is given
        if metadata.is_required \
                and metadata.default is not None \
                and not metadata.value_was_explicitly_changed:
            self.modify_parser_args(metadata.name, str(metadata.default))

        # set default value if is_constant
        if metadata.is_constant:
            self.modify_parser_args(metadata.name, str(metadata.default))
            metadata.value_was_explicitly_changed = False

        # get options
        self.options, _ = self.parser.parse_known_args(self.parser_args)

    def set_option_value(self, name, value):
        self.modify_parser_args(name, value)
        self.options, _ = self.parser.parse_known_args(self.parser_args)

    def modify_parser_args(self, key, value):
        key = '--{}'.format(key)
        if key in self.parser_args:
            index = self.parser_args.index(key) + 1
            self.parser_args[index] = str(value)
        else:
            self.parser_args.extend([key, str(value)])
