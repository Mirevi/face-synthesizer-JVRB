import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch, call

from config import *


class TestEnum(Enum):
    a = 'a'
    b = 'b'
    c = 'c'

    def __str__(self):
        return self.value


class ConfigOptionPackageRequiredImpl(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'coat_color', 'black', 'Color of the coat.', False),
        ]


class ConfigOptionPackageConditionalRequiredImpl1(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'dog_size', 'Small', 'Size of the dog.', False),
        ]


class ConfigOptionPackageConditionalRequiredImpl2(ConfigOptionPackage):
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'toy', 'Bone', 'Toy', False),
        ]


class ConfigOptionPackageImpl(COPWithModifiableDefaults):
    options_metadata = [
        ConfigOptionMetadata(str, 'name', 'Charly', 'Name.', False),
        ConfigOptionMetadata(str, 'type', 'dog', 'The animal type.', False),
        ConfigOptionMetadata(int, 'age', 10, 'Age', False),
        ConfigOptionMetadata(bool, 'has_toy', True, 'If there is a toy', False),
    ]

    @staticmethod
    def get_options_metadata() -> list:
        return ConfigOptionPackageImpl.options_metadata

    conditional_options_metadata = [
        ConfigOptionMetadata(str, 'dog_breed', 'jack russel', 'The breed of the dog', False)
    ]

    @staticmethod
    def get_conditional_options_metadata(options) -> list:
        if options.type == 'dog':
            return ConfigOptionPackageImpl.conditional_options_metadata
        else:
            return []

    default_modifications = [
        ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'coat_color', 'gray')
    ]

    @staticmethod
    def get_default_modifications() -> list:
        return ConfigOptionPackageImpl.default_modifications

    conditional_default_modifications = [
        ConfigDefaultModification(ConfigOptionPackageConditionalRequiredImpl1, 'dog_size', 'Medium')
    ]

    @staticmethod
    def get_conditional_default_modifications(options):
        if options.type == 'dog':
            return ConfigOptionPackageImpl.conditional_default_modifications
        else:
            return []


class WrongConfigOptionPackageImpl:
    @staticmethod
    def get_options_metadata() -> list:
        return [
            ConfigOptionMetadata(str, 'name', 'Charly', 'Name.', False),
            ConfigOptionMetadata(str, 'type', 'dog', 'The animal type.', False),
            ConfigOptionMetadata(int, 'age', 10, 'Age', False),
        ]


class ConfigPackageProviderImpl(ConfigPackageProvider):
    @staticmethod
    def get_required_option_packages() -> list:
        return [ConfigOptionPackageImpl]

    @staticmethod
    def get_conditional_option_packages(options) -> list:
        if options.has_toy:
            return [ConfigOptionPackageConditionalRequiredImpl2]
        else:
            return []


class WrongConfigPackageProviderImpl:
    @staticmethod
    def get_required_option_packages() -> list:
        return [ConfigOptionPackageImpl]


class ConfigOptionMetadataTests(TestCase):
    def test_initDefaultDatatypeValidation_correctType_noError(self):
        try:
            ConfigOptionMetadata(str, '', 'string', 'help', False)
            ConfigOptionMetadata(int, '', 5, 'help', True)
            ConfigOptionMetadata(float, '', 3.0, 'help', False)
            ConfigOptionMetadata(TestEnum, '', TestEnum.a, 'help', True, list(TestEnum))
        except TypeError:
            self.fail('Unexpected TypeError')

    def test_initDefaultDatatypeValidation_wrongType_raiseTypeError(self):
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(str, '', 5, 'help', False)
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(int, '', 'test', 'help', True)
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(float, '', TestEnum.a, 'help', False)
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(TestEnum, '', 3.0, 'help', True)

    def test_initChoicesValidation_correctTypes_noError(self):
        try:
            ConfigOptionMetadata(str, '', 'string', 'help', False, choices=['string', 'test'])
            ConfigOptionMetadata(int, '', 5, 'help', True, choices=range(0, 10))
            ConfigOptionMetadata(float, '', 3.0, 'help', False, choices=[1.0, 1.1, 3.0])
            ConfigOptionMetadata(TestEnum, '', TestEnum.a, 'help', True, choices=list(TestEnum))
        except TypeError:
            self.fail('Unexpected TypeError')

    def test_initChoicesValidation_wrongType_raiseTypeError(self):
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(str, '', 'string', 'help', False, choices=['string', 0])
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(int, '', 5, 'help', True, choices=[1, 2, 'test'])
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(float, '', 3.0, 'help', False, choices=[1.0, 1.1, '3.0'])
        with self.assertRaises(TypeError):
            ConfigOptionMetadata(TestEnum, '', TestEnum.a, 'help', True, choices=['test'])

    def test_initDefaultInChoiceValidation_isInChoice_noError(self):
        try:
            ConfigOptionMetadata(str, '', 'test', 'help', False, choices=['string', 'test'])
            ConfigOptionMetadata(int, '', 5, 'help', True, choices=[1, 5, 4])
            ConfigOptionMetadata(float, '', 3.0, 'help', False, choices=[1.0, 1.1, 3.0])
            ConfigOptionMetadata(TestEnum, '', TestEnum.a, 'help', True, choices=[TestEnum.a, TestEnum.b])
        except TypeError:
            self.fail('Unexpected TypeError')

    def test_initDefaultInChoiceValidation_isNotInChoice_raiseValueError(self):
        with self.assertRaises(ValueError):
            ConfigOptionMetadata(str, '', 'default', 'help', False, choices=['string', 'test'])
        with self.assertRaises(ValueError):
            ConfigOptionMetadata(int, '', 75, 'help', True, choices=[1, 5, 4])
        with self.assertRaises(ValueError):
            ConfigOptionMetadata(float, '', 4.0, 'help', False, choices=[1.0, 1.1, 3.0])
        with self.assertRaises(ValueError):
            ConfigOptionMetadata(TestEnum, '', TestEnum.c, 'help', True, choices=[TestEnum.a, TestEnum.b])


class ConfigDefaultModificationTests(TestCase):
    def test_initPackageValidation_correctType_noError(self):
        try:
            ConfigDefaultModification(ConfigOptionPackageImpl, 'name', 'value')
            ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'name', 'value')
            ConfigDefaultModification(ConfigOptionPackageConditionalRequiredImpl1, 'name', 'value')
        except TypeError:
            self.fail('Unexpected TypeError')

    def test_initPackageValidation_wrongType_raiseTypeError(self):
        with self.assertRaises(TypeError):
            ConfigDefaultModification(WrongConfigOptionPackageImpl, 'name', 'value')


class BaseConfigProviderTests(TestCase):
    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addPackageProviders_correctTypesInList_noError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()

        try:
            config.add_package_providers([ConfigPackageProviderImpl, ConfigPackageProviderImpl])
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addPackageProviders_wrongTypeInList_raiseTypeError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()

        with self.assertRaises(TypeError):
            config.add_package_providers([ConfigPackageProviderImpl, WrongConfigPackageProviderImpl])
        with self.assertRaises(TypeError):
            config.add_package_providers([ConfigPackageProviderImpl, ConfigOptionPackageImpl])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addPackageProvider_correctType_noError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()

        try:
            config.add_package_provider(ConfigPackageProviderImpl)
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addPackageProvider_wrongType_raiseTypeError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()

        with self.assertRaises(TypeError):
            config.add_package_provider(WrongConfigPackageProviderImpl)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addPackageProvider_doubleRegistration_onlyOnceInConfig(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()

        config.add_package_provider(ConfigPackageProviderImpl)
        config.add_package_provider(ConfigPackageProviderImpl)

        self.assertIs(config.package_provider.count(ConfigPackageProviderImpl), 1)
        config.add_option_packages.assert_called_with(ConfigPackageProviderImpl.get_required_option_packages())

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_conditionalsMet_addedConditionals(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_package_provider(ConfigPackageProviderImpl)
        config.options.has_toy = True

        config.gather_options()

        calls = [call([ConfigOptionPackageImpl]), call([ConfigOptionPackageConditionalRequiredImpl2])]
        config.add_option_packages.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_conditionalsNotMet_conditionalsNotAdded(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_package_provider(ConfigPackageProviderImpl)
        config.options.has_toy = False

        config.gather_options()

        calls = [call([ConfigOptionPackageImpl])]
        config.add_option_packages.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_conditionalsNotAvailable_raiseAttributeError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_package_provider(ConfigPackageProviderImpl)

        with self.assertRaises(AttributeError):
            config.gather_options()

        calls = [call([ConfigOptionPackageImpl])]
        config.add_option_packages.assert_has_calls(calls)


class BaseConfigPackageTests(TestCase):
    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackages_correctTypesInList_noError(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        try:
            config.add_option_packages([ConfigOptionPackageImpl, ConfigOptionPackageConditionalRequiredImpl2])
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackages_wrongTypeInList_raiseTypeError(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        with self.assertRaises(TypeError):
            config.add_option_packages([ConfigOptionPackageImpl, WrongConfigOptionPackageImpl])
        with self.assertRaises(TypeError):
            config.add_option_packages([ConfigOptionPackageImpl, ConfigPackageProviderImpl])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackage_correctTypesInList_noError(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        try:
            config.add_option_package(ConfigOptionPackageImpl)
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackage_wrongType_raiseTypeError(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        with self.assertRaises(TypeError):
            config.add_option_package(WrongConfigOptionPackageImpl)
        with self.assertRaises(TypeError):
            config.add_option_package(ConfigPackageProviderImpl)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackage_doubleRegistration_onlyOnceInConfig(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        config.add_option_package(ConfigOptionPackageImpl)
        config.add_option_package(ConfigOptionPackageImpl)

        self.assertIs(config.option_packages.count(ConfigOptionPackageImpl), 1)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackage_withModifications_necessaryPackagesAdded(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        config.add_option_package(ConfigOptionPackageImpl)

        self.assertIs(len(config.option_packages), 2)
        self.assertTrue(ConfigOptionPackageImpl in config.option_packages)
        self.assertTrue(ConfigOptionPackageRequiredImpl in config.option_packages)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addOptionPackage_withModifications_defaultsModified(self):
        config = BaseConfig()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()

        config.add_option_package(ConfigOptionPackageImpl)

        config.apply_default_modifications.assert_called_with(ConfigOptionPackageImpl.default_modifications)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_optionConditionalsMet_addedConditionalOptions(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'dog'

        config.gather_options()

        calls = [call(ConfigOptionPackageImpl.conditional_options_metadata)]
        config.add_multiple_option_metadata.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_optionConditionalsNotMet_conditionalOptionsNotAdded(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'cat'

        config.gather_options()

        calls = [call([])]
        config.add_multiple_option_metadata.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_optionConditionalsNotAvailable_raiseAttributeError(self):
        config = BaseConfig()
        config.add_option_packages = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]

        with self.assertRaises(AttributeError):
            config.gather_options()

        config.add_multiple_option_metadata.assert_not_called()

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsMet_necessaryPackagesAdded(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'dog'

        config.gather_options()

        calls = [call(ConfigOptionPackageConditionalRequiredImpl1)]
        config.add_option_package.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsMet_defaultsModified(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'dog'

        config.gather_options()

        calls = [call(ConfigOptionPackageImpl.conditional_default_modifications)]
        config.apply_default_modifications.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsNotMet_noPackagesAdded(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'cat'

        config.gather_options()

        config.add_option_package.assert_not_called()

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsNotMet_noDefaultsModified(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]
        config.options.type = 'cat'

        config.gather_options()

        calls = [call([])]
        config.apply_default_modifications.assert_has_calls(calls)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsNotAvailable_raiseAttributeError(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]

        with self.assertRaises(AttributeError):
            config.gather_options()

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_modificationConditionalsNotAvailable_raiseAttributeError(self):
        config = BaseConfig()
        config.add_option_package = MagicMock()
        config.add_multiple_option_metadata = MagicMock()
        config.apply_default_modifications = MagicMock()
        config.option_packages = [ConfigOptionPackageImpl]

        with self.assertRaises(AttributeError):
            config.gather_options()


class BaseConfigOptionTests(TestCase):
    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMultipleMetadata_correctTypesInList_noError(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()

        try:
            config.add_multiple_option_metadata([
                ConfigOptionMetadata(str, 'name', 'default', 'help', False),
                ConfigOptionMetadata(int, 'name1', 1, 'help', False),
                ConfigOptionMetadata(str, 'name2', None, 'help', True),
            ])
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMultipleMetadata_wrongTypesInList_raiseTypeError(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()

        with self.assertRaises(TypeError):
            config.add_multiple_option_metadata([
                ConfigOptionMetadata(str, 'name', 'default', 'help', False),
                ConfigOptionMetadata(int, 'name1', 1, 'help', False),
                ConfigDefaultModification(None, 'name', 'value'),
            ])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_correctType_noError(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()

        try:
            config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', False))
            config.add_option_metadata(ConfigOptionMetadata(int, 'name1', 1, 'help', False))
            config.add_option_metadata(ConfigOptionMetadata(str, 'name2', 'default2', 'help', True))
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_correctType_metadataInConfig(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()

        metadata = ConfigOptionMetadata(str, 'name', 'default', 'help', False)
        config.add_option_metadata(metadata)

        self.assertTrue(metadata in config.options_metadata.values())
        self.assertEqual(metadata, config.options_metadata[metadata.name])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_correctType_createAndAddOptionCalled(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()
        metadata = ConfigOptionMetadata(str, 'name', 'default', 'help', False)

        config.add_option_metadata(metadata)

        config.create_and_add_option.assert_called_with(metadata)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_choiceIsInvalidDtype_raiseTypeError(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()
        metadata = ConfigOptionMetadata(str, 'name', 'default1', 'help', False)
        metadata.choices = ['default1', 0]

        with self.assertRaises(TypeError):
            config.add_option_metadata(metadata)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_defaultNotInChoices_raiseValueError(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()
        metadata = ConfigOptionMetadata(str, 'name', 'no choice', 'help', False)
        metadata.choices = ['default1', 'default2']

        with self.assertRaises(ValueError):
            config.add_option_metadata(metadata)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_defaultInChoices_metadataInConfig(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()
        metadata = ConfigOptionMetadata(str, 'name', 'default1', 'help', False, choices=['default1', 'default2'])

        config.add_option_metadata(metadata)

        self.assertTrue(metadata in config.options_metadata.values())
        self.assertEqual(metadata, config.options_metadata[metadata.name])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_addMetadata_defaultInChoices_createAndAddOptionCalled(self):
        config = BaseConfig()
        config.create_and_add_option = MagicMock()
        metadata = ConfigOptionMetadata(str, 'name', 'default1', 'help', False, choices=['default1', 'default2'])

        config.add_option_metadata(metadata)

        config.create_and_add_option.assert_called_with(metadata)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_requiredIsGiven_noError(self):
        config = BaseConfig()
        config.options_metadata = {'name': ConfigOptionMetadata(str, 'name', None, 'help', True)}
        config.options.name = 'test'

        try:
            config.gather_options()
        except RuntimeError:
            self.fail('Unexpected RuntimeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_gatherOptions_requiredIsNone_raiseRuntimeError(self):
        config = BaseConfig()
        config.options_metadata = {'name': ConfigOptionMetadata(str, 'name', None, 'help', True)}
        config.options.name = None

        with self.assertRaises(RuntimeError):
            config.gather_options()


class BaseConfigModificationTests(TestCase):
    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModifications_correctTypesInList_noError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        metadata = [
            ConfigOptionMetadata(str, 'name1', 'default1', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', 'default3', 'help', True),
        ]
        config.add_multiple_option_metadata(metadata)
        config.add_option_package(ConfigOptionPackageRequiredImpl)

        try:
            config.apply_default_modifications([
                ConfigDefaultModification(None, 'name1', 'new default 1'),
                ConfigDefaultModification(None, 'name2', 0),
                ConfigDefaultModification(None, 'name3', 'new default 3'),
                ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'coat_color', 'red'),
            ])
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModifications_wrongTypesInList_raiseTypeError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        metadata = [
            ConfigOptionMetadata(str, 'name1', 'default1', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', 'default3', 'help', True),
        ]
        config.add_multiple_option_metadata(metadata)
        config.add_option_package(ConfigOptionPackageRequiredImpl)

        with self.assertRaises(TypeError):
            config.apply_default_modifications([
                ConfigDefaultModification(None, 'name1', 'new default 1'),
                ConfigDefaultModification(None, 'name2', 0),
                ConfigDefaultModification(None, 'name3', 'new default 3'),
                ConfigOptionMetadata(str, 'name4', 'default4', 'help', False),
            ])

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_correctType_noError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        metadata = [
            ConfigOptionMetadata(str, 'name1', 'default1', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', 'default3', 'help', True),
        ]
        config.add_multiple_option_metadata(metadata)
        config.add_option_package(ConfigOptionPackageRequiredImpl)

        try:
            config.apply_default_modification(ConfigDefaultModification(None, 'name1', 'new default 1'))
            config.apply_default_modification(ConfigDefaultModification(None, 'name2', 0))
            config.apply_default_modification(ConfigDefaultModification(None, 'name3', 'new default 3'))
            config.apply_default_modification(
                ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'coat_color', 'red'))
        except TypeError:
            self.fail('Unexpected TypeError')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_packageNotInConfig_raiseValueError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()

        with self.assertRaises(ValueError):
            config.apply_default_modification(
                ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'coat_color', 'red'))

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_nameNotInPackage_raiseValueError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        config.add_option_package(ConfigOptionPackageRequiredImpl)

        with self.assertRaises(ValueError):
            config.apply_default_modification(
                ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'color', 'red'))

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_nameNotInConfig_raiseValueError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()

        with self.assertRaises(ValueError):
            config.apply_default_modification(ConfigDefaultModification(None, 'name1', 'new default 1'))

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_wrongDtype_raiseTypeError(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', False))

        with self.assertRaises(TypeError):
            config.apply_default_modification(ConfigDefaultModification(None, 'name', 0))

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_valid_defaultChanged(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        metadata = [
            ConfigOptionMetadata(str, 'name1', 'default1', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
        ]
        config.add_multiple_option_metadata(metadata)
        config.add_option_package(ConfigOptionPackageRequiredImpl)

        config.apply_default_modification(ConfigDefaultModification(None, 'name1', 'new default 1'))
        config.apply_default_modification(ConfigDefaultModification(None, 'name2', 0))
        config.apply_default_modification(ConfigDefaultModification(None, 'name3', 'new default 3'))
        config.apply_default_modification(
            ConfigDefaultModification(ConfigOptionPackageRequiredImpl, 'coat_color', 'red'))

        self.assertEqual(config.options_metadata['name1'].default, 'new default 1')
        self.assertEqual(config.options_metadata['name2'].default, 0)
        self.assertEqual(config.options_metadata['name3'].default, 'new default 3')
        self.assertEqual(config.options_metadata['coat_color'].default, 'red')

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_requiredNotExplicitlyChanged_valueChanged(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', None, 'help', True))

        config.apply_default_modification(ConfigDefaultModification(None, 'name', 'default'))

        config.set_option_value.assert_called_with('name', 'default')
        self.assertIs(config.options_metadata["name"].value_was_explicitly_changed, False)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_applyDefaultModification_requiredExplicitlyChanged_valueNotChanged(self):
        config = BaseConfig()
        config.set_option_value = MagicMock()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', None, 'help', True))
        config.options_metadata["name"].value_was_explicitly_changed = True

        config.apply_default_modification(ConfigDefaultModification(None, 'name', 'default'))

        config.set_option_value.assert_not_called()
        self.assertIs(config.options_metadata["name"].value_was_explicitly_changed, True)


class BaseConfigUtilityTest(TestCase):
    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_str_emptyConfig_stringEmpty(self):
        config = BaseConfig()
        expected = ''

        actual = str(config)

        self.assertEqual(expected, actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_str_fullConfig_stringAsExpected(self):
        config = BaseConfig()
        config.options.string = 'str'
        config.options.integer = 0
        config.options.float = 1.0
        config.options.enum = TestEnum.a

        expected = '{:>25}: {:<30}\n'.format('enum', str(TestEnum.a)) \
                   + '{:>25}: {:<30}\n'.format('float', str(1.0)) \
                   + '{:>25}: {:<30}\n'.format('integer', str(0)) \
                   + '{:>25}: {:<30}'.format('string', str('str'))

        actual = str(config)

        self.assertEqual(expected, actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withProviderClass_returnTrue(self):
        config = BaseConfig()
        config.package_provider.append(ConfigPackageProviderImpl)

        try:
            actual = ConfigPackageProviderImpl in config
        except:
            self.fail('Unexpected Error occured')

        self.assertTrue(actual)

        try:
            actual = ConfigPackageProviderImpl() in config
        except:
            self.fail('Unexpected Error occured')

        self.assertTrue(actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withoutProviderClass_returnFalse(self):
        config = BaseConfig()

        try:
            actual = ConfigPackageProviderImpl in config
        except:
            self.fail('Unexpected Error occured')

        self.assertFalse(actual)

        try:
            actual = ConfigPackageProviderImpl() in config
        except:
            self.fail('Unexpected Error occured')

        self.assertFalse(actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withPackageClass_returnTrue(self):
        config = BaseConfig()
        config.option_packages.append(ConfigOptionPackageImpl)

        try:
            actual = ConfigOptionPackageImpl in config
        except:
            self.fail('Unexpected Error occured')

        self.assertTrue(actual)

        try:
            actual = ConfigOptionPackageImpl() in config
        except:
            self.fail('Unexpected Error occured')

        self.assertTrue(actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withoutPackageClass_returnFalse(self):
        config = BaseConfig()

        try:
            actual = ConfigOptionPackageImpl in config
        except:
            self.fail('Unexpected Error occured')

        self.assertFalse(actual)

        try:
            actual = ConfigOptionPackageImpl() in config
        except:
            self.fail('Unexpected Error occured')

        self.assertFalse(actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withMetadataInstance_returnTrue(self):
        config = BaseConfig()
        config.options_metadata = {'name', ConfigOptionMetadata(str, 'name', 'default', 'help')}

        try:
            actual = 'name' in config
        except:
            self.fail('Unexpected Error occured')

        self.assertTrue(actual)

    @patch.object(BaseConfig, '__abstractmethods__', set())
    def test_contains_withoutMetadataInstance_returnFalse(self):
        config = BaseConfig()

        try:
            actual = 'name' in config
        except:
            self.fail('Unexpected Error occured')

        self.assertFalse(actual)


class StandardConfigTests(TestCase):
    def test_createAndAddOption_notRequired_addedToOptions(self):
        config = StandardConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', False))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'default')

    def test_createAndAddOption_requiredWithExplicitNoDefault_addedToOptionsWithExplicitValue(self):
        config = StandardConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', None, 'help', True))
        config['name'] = 'required'

        self.assertEqual(len(vars(config.options)), 1)
        self.assertTrue(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'required')

    def test_createAndAddOption_requiredNoExplicitWithDefault_addedToOptionsWithDefaultValue(self):
        config = StandardConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', True))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'default')

    def test_createAndAddOption_requiredWithExplicitWithDefault_addedToOptionsWithExplicitValue(self):
        config = StandardConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', True))
        config['name'] = 'required'

        self.assertEqual(len(vars(config.options)), 1)
        self.assertTrue(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'required')

    def test_createAndAddOption_enumAttribute_addedToOptions(self):
        config = StandardConfig()

        config.add_option_metadata(ConfigOptionMetadata(TestEnum, 'name', TestEnum.a, 'help', False, list(TestEnum)))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, TestEnum.a)

    def test_createAndAddOption_withArguments_valueFromArguments(self):
        config = StandardConfig({'name': TestEnum.c})

        config.add_option_metadata(ConfigOptionMetadata(TestEnum, 'name', TestEnum.a, 'help', False, list(TestEnum)))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertTrue(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, TestEnum.c)

    def test_setitem_wrongType_raiseTypeError(self):
        config = StandardConfig()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help'))

        with self.assertRaises(TypeError):
            config['name'] = 0

    def test_setitem_notInChoices_raiseValueError(self):
        config = StandardConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default1', 'help', choices=['default1', 'default2']))

        with self.assertRaises(ValueError):
            config['name'] = 'test'

    def test_setitem_correctType_valueSet(self):
        config = StandardConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default', 'help'))

        config['name'] = 'test'

        self.assertEqual(config['name'], 'test')

    def test_setitem_inChoices_valueSet(self):
        config = StandardConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default1', 'help', choices=['default1', 'default2']))

        config['name'] = 'default1'

        self.assertEqual(config['name'], 'default1')

    def test_setitem_isConstant_raiseAttributeError(self):
        config = StandardConfig()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default1', 'help', is_constant=True))

        with self.assertRaises(AttributeError):
            config['name'] = 'default1'

    def test_valuesFromString_emptyString_raiseValueError(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = ''

        with self.assertRaises(ValueError):
            config.set_values_from_string(s)

    def test_valuesFromString_emptyStringIgnoreUnexpected_noErrorAndNothingChanged(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = ''

        try:
            config.set_values_from_string(s, True)
        except:
            self.fail('Unexpected Error')

        self.assertIs(len(vars(config.options)), 4)
        self.assertFalse(config.options_metadata['name1'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name2'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name3'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name4'].value_was_explicitly_changed)
        self.assertEqual(config.options.name1, 'default')
        self.assertEqual(config.options.name2, 1)
        self.assertEqual(config.options.name3, None)
        self.assertEqual(config.options.name4, TestEnum.a)

    def test_valuesFromString_unexpectedAttribute_raiseValueError(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = '{:>25}: {:<30}\n'.format('unexpected', 'test')

        with self.assertRaises(ValueError):
            config.set_values_from_string(s)

    def test_valuesFromString_unexpectedAttributeIgnored_noErrorAndNothingChanged(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = '{:>25}: {:<30}\n'.format('unexpected', 'test')

        try:
            config.set_values_from_string(s, True)
        except:
            self.fail('Unexpected Error')

        self.assertIs(len(vars(config.options)), 4)
        self.assertFalse(config.options_metadata['name1'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name2'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name3'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name4'].value_was_explicitly_changed)
        self.assertEqual(config.options.name1, 'default')
        self.assertEqual(config.options.name2, 1)
        self.assertEqual(config.options.name3, None)
        self.assertEqual(config.options.name4, TestEnum.a)

    def test_valuesFromString_oneAttribute_onlyOneValueChanged(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = '{:>25}: {:<30}'.format('name1', 'test')

        try:
            config.set_values_from_string(s)
        except:
            self.fail('Unexpected Error')

        self.assertIs(len(vars(config.options)), 4)
        self.assertTrue(config.options_metadata['name1'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name2'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name3'].value_was_explicitly_changed)
        self.assertFalse(config.options_metadata['name4'].value_was_explicitly_changed)
        self.assertEqual(config.options.name1, 'test')
        self.assertEqual(config.options.name2, 1)
        self.assertEqual(config.options.name3, None)
        self.assertEqual(config.options.name4, TestEnum.a)

    def test_valuesFromString_allAttributes_allValuesChanged(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = '' \
            + '{:>25}: {:<30}\n'.format('name1', 'test') \
            + '{:>25}: {:<30}\n'.format('name2', str(0)) \
            + '{:>25}: {:<30}\n'.format('name3', 'not None') \
            + '{:>25}: {:<30}'.format('name4', str(TestEnum.b))

        try:
            config.set_values_from_string(s)
        except:
            self.fail('Unexpected Error')

        self.assertIs(len(vars(config.options)), 4)
        self.assertTrue(config.options_metadata['name1'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name2'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name3'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name4'].value_was_explicitly_changed)
        self.assertEqual(config.options.name1, 'test')
        self.assertEqual(config.options.name2, 0)
        self.assertEqual(config.options.name3, 'not None')
        self.assertEqual(config.options.name4, TestEnum.b)

    def test_valuesFromString_strippedStringAllAttributes_allValuesChanged(self):
        config = StandardConfig()
        config.add_multiple_option_metadata([
            ConfigOptionMetadata(str, 'name1', 'default', 'help', False),
            ConfigOptionMetadata(int, 'name2', 1, 'help', False),
            ConfigOptionMetadata(str, 'name3', None, 'help', True),
            ConfigOptionMetadata(TestEnum, 'name4', TestEnum.a, 'help', False, list(TestEnum)),
        ])
        s = 'name1: test\nname2: 0\nname3: not None\nname4: {}'.format(str(TestEnum.b))

        try:
            config.set_values_from_string(s)
        except:
            self.fail('Unexpected Error')

        self.assertIs(len(vars(config.options)), 4)
        self.assertTrue(config.options_metadata['name1'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name2'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name3'].value_was_explicitly_changed)
        self.assertTrue(config.options_metadata['name4'].value_was_explicitly_changed)
        self.assertEqual(config.options.name1, 'test')
        self.assertEqual(config.options.name2, 0)
        self.assertEqual(config.options.name3, 'not None')
        self.assertEqual(config.options.name4, TestEnum.b)


class ArgparseConfigTests(TestCase):
    def test_createAndAddOption_notRequired_addedToOptions(self):
        config = ArgparseConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', False))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'default')

    def test_createAndAddOption_requiredWithExplicitNoDefault_addedToOptionsWithExplicitValue(self):
        config = ArgparseConfig()
        config.parser_args = ['--name', 'required']

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', None, 'help', True))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertTrue(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'required')

    def test_createAndAddOption_requiredNoExplicitWithDefault_addedToOptionsWithDefaultValue(self):
        config = ArgparseConfig()

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', True))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'default')

    def test_createAndAddOption_requiredWithExplicitWithDefault_addedToOptionsWithExplicitValue(self):
        config = ArgparseConfig()
        config.parser_args = ['--name', 'required']

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', True))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertTrue(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'required')

    def test_createAndAddOption_enumAttribute_addedToOptions(self):
        config = ArgparseConfig()

        config.add_option_metadata(ConfigOptionMetadata(TestEnum, 'name', TestEnum.a, 'help', False, list(TestEnum)))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, TestEnum.a)

    def test_createAndAddOption_constantOption_seDefaultAlthoughExplicitIsGiven(self):
        config = ArgparseConfig()
        config.parser_args = ['--name', 'modification']

        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default', 'help', is_constant=True))

        self.assertEqual(len(vars(config.options)), 1)
        self.assertFalse(config.options_metadata['name'].value_was_explicitly_changed)
        self.assertEqual(config.options.name, 'default')

    def test_setitem_wrongType_raiseTypeError(self):
        config = ArgparseConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default', 'help'))

        with self.assertRaises(TypeError):
            config['name'] = 0

    def test_setitem_notInChoices_raiseValueError(self):
        config = ArgparseConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default1', 'help', choices=['default1', 'default2']))

        with self.assertRaises(ValueError):
            config['name'] = 'test'

    def test_setitem_correctType_valueSet(self):
        config = ArgparseConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default', 'help'))

        config['name'] = 'test'

        self.assertEqual(config['name'], 'test')

    def test_setitem_inChoices_valueSet(self):
        config = ArgparseConfig()
        config.add_option_metadata(
            ConfigOptionMetadata(str, 'name', 'default1', 'help', choices=['default1', 'default2']))

        config['name'] = 'default1'

        self.assertEqual(config['name'], 'default1')

    def test_setitem_isConstant_raiseAttributeError(self):
        config = ArgparseConfig()
        config.add_option_metadata(ConfigOptionMetadata(str, 'name', 'default1', 'help', is_constant=True))

        with self.assertRaises(AttributeError):
            config['name'] = 'default1'


if __name__ == '__main__':
    unittest.main()
