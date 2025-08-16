"""
Simple test cases for enterprise integrations (syntax and basic functionality)
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEnterpriseIntegrationsSyntax(unittest.TestCase):
    """Test basic syntax and structure of enterprise integrations"""

    def test_identity_management_syntax(self):
        """Test that identity management file has correct syntax"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        # Parse the source code - will raise SyntaxError if invalid
        tree = ast.parse(source)

        # Check that main classes are defined
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        expected_classes = [
            'ServiceNowIntegration',
            'SailPointIntegration',
            'MicrosoftEntraIntegration',
            'BritiveIntegration',
        ]

        for expected_class in expected_classes:
            self.assertIn(expected_class, class_names, f"Class {expected_class} not found")

    def test_teams_communication_syntax(self):
        """Test that Teams communication file has correct syntax"""
        import ast

        teams_file = Path(__file__).parent.parent / "integrations" / "productivity" / "teams_communication.py"

        with open(teams_file, 'r') as f:
            source = f.read()

        # Parse the source code - will raise SyntaxError if invalid
        tree = ast.parse(source)

        # Check that main classes are defined
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        expected_classes = ['MicrosoftTeamsIntegration', 'TeamsMessageBuilder']

        for expected_class in expected_classes:
            self.assertIn(expected_class, class_names, f"Class {expected_class} not found")

    def test_base_integration_inheritance(self):
        """Test that enterprise integrations inherit from BaseIntegration"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Find class definitions and their base classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.endswith('Integration'):
                    # Check that it inherits from BaseIntegration
                    base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
                    self.assertIn('BaseIntegration', base_names, f"{node.name} does not inherit from BaseIntegration")

    def test_required_methods_present(self):
        """Test that required abstract methods are implemented"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Required methods from BaseIntegration
        required_methods = ['initialize', 'cleanup', 'health_check', '_execute_operation_impl']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith('Integration'):
                # Include both regular and async function definitions
                class_methods = [
                    method.name for method in node.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                for required_method in required_methods:
                    self.assertIn(
                        required_method, class_methods, f"{node.name} missing required method {required_method}"
                    )

    def test_configuration_keys_defined(self):
        """Test that REQUIRED_CONFIG_KEYS are defined where needed"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Classes that should have REQUIRED_CONFIG_KEYS
        classes_with_required_keys = [
            'ServiceNowIntegration',
            'SailPointIntegration',
            'MicrosoftEntraIntegration',
            'BritiveIntegration',
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in classes_with_required_keys:
                class_attributes = [
                    attr.targets[0].id
                    for attr in node.body
                    if isinstance(attr, ast.Assign) and isinstance(attr.targets[0], ast.Name)
                ]

                self.assertIn('REQUIRED_CONFIG_KEYS', class_attributes, f"{node.name} missing REQUIRED_CONFIG_KEYS")

    def test_teams_message_builder_methods(self):
        """Test that TeamsMessageBuilder has expected methods"""
        import ast

        teams_file = Path(__file__).parent.parent / "integrations" / "productivity" / "teams_communication.py"

        with open(teams_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Find TeamsMessageBuilder class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'TeamsMessageBuilder':
                methods = [method.name for method in node.body if isinstance(method, ast.FunctionDef)]

                expected_methods = [
                    'set_title',
                    'set_summary',
                    'set_theme_color',
                    'add_section',
                    'add_fact',
                    'add_action',
                    'build',
                ]

                for expected_method in expected_methods:
                    self.assertIn(expected_method, methods, f"TeamsMessageBuilder missing method {expected_method}")
                break
        else:
            self.fail("TeamsMessageBuilder class not found")

    def test_convenience_functions_defined(self):
        """Test that convenience functions are defined"""
        import ast

        teams_file = Path(__file__).parent.parent / "integrations" / "productivity" / "teams_communication.py"

        with open(teams_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Check for convenience functions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        expected_functions = ['create_simple_message', 'create_alert_card', 'create_status_card']

        for expected_function in expected_functions:
            self.assertIn(expected_function, functions, f"Convenience function {expected_function} not found")

    def test_mcp_tool_registration_syntax(self):
        """Test that MCP tool registrations in wand.py have correct syntax"""
        import ast

        wand_file = Path(__file__).parent.parent / "wand.py"

        with open(wand_file, 'r') as f:
            source = f.read()

        # Parse the source code - will raise SyntaxError if invalid
        tree = ast.parse(source)

        # Check that our new tool functions are defined (look in nested functions within create_orchestrator_stdio_server)
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)

        expected_tools = ['servicenow', 'sailpoint', 'entra', 'britive', 'teams']

        for expected_tool in expected_tools:
            self.assertIn(expected_tool, functions, f"MCP tool function {expected_tool} not found in wand.py")

    def test_requirements_files_updated(self):
        """Test that requirements files include new dependencies"""

        # Check requirements-integrations.txt
        integrations_req_file = Path(__file__).parent.parent / "requirements-integrations.txt"

        with open(integrations_req_file, 'r') as f:
            content = f.read()

        expected_packages = ['pysnc', 'msal', 'msgraph-sdk', 'azure-identity', 'britive', 'pymsteams']

        for package in expected_packages:
            self.assertIn(package, content, f"Package {package} not found in requirements-integrations.txt")

        # Check pyproject.toml
        pyproject_file = Path(__file__).parent.parent / "pyproject.toml"

        with open(pyproject_file, 'r') as f:
            content = f.read()

        for package in expected_packages:
            self.assertIn(package, content, f"Package {package} not found in pyproject.toml")

    def test_documentation_exists(self):
        """Test that documentation files exist"""

        # Check enterprise integrations documentation
        docs_file = Path(__file__).parent.parent / "docs" / "ENTERPRISE_INTEGRATIONS.md"
        self.assertTrue(docs_file.exists(), "ENTERPRISE_INTEGRATIONS.md documentation not found")

        with open(docs_file, 'r') as f:
            content = f.read()

        # Check that it covers all integrations
        expected_sections = ['ServiceNow', 'SailPoint', 'Microsoft Entra', 'Britive', 'Microsoft Teams']

        for section in expected_sections:
            self.assertIn(section, content, f"Documentation missing section for {section}")

    def test_error_handling_patterns(self):
        """Test that error handling patterns are consistent"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Check that _execute_operation_impl methods have try/except blocks
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == '_execute_operation_impl':
                # Check for try/except in the method
                has_try_except = any(isinstance(child, ast.Try) for child in ast.walk(node))
                self.assertTrue(has_try_except, f"_execute_operation_impl method should have try/except blocks")

    def test_async_method_signatures(self):
        """Test that async methods are properly defined"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Methods that should be async
        async_methods = ['initialize', 'cleanup', 'health_check', '_execute_operation_impl']

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith('Integration'):
                for method in node.body:
                    if isinstance(method, ast.AsyncFunctionDef) and method.name in async_methods:
                        # Found async method - this is good
                        continue
                    elif isinstance(method, ast.FunctionDef) and method.name in async_methods:
                        self.fail(f"Method {method.name} in {node.name} should be async")


class TestEnterpriseIntegrationLogic(unittest.TestCase):
    """Test logical structure and patterns in enterprise integrations"""

    def test_config_environment_variable_patterns(self):
        """Test that environment variables follow consistent naming patterns"""

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            content = f.read()

        # Check for consistent environment variable naming
        env_var_patterns = [
            'SERVICENOW_INSTANCE_URL',
            'SERVICENOW_USERNAME',
            'SERVICENOW_PASSWORD',
            'SAILPOINT_BASE_URL',
            'SAILPOINT_CLIENT_ID',
            'SAILPOINT_CLIENT_SECRET',
            'AZURE_TENANT_ID',
            'AZURE_CLIENT_ID',
            'AZURE_CLIENT_SECRET',
            'BRITIVE_TENANT',
            'BRITIVE_API_TOKEN',
        ]

        for env_var in env_var_patterns:
            self.assertIn(env_var, content, f"Environment variable {env_var} not found")

    def test_operation_methods_naming(self):
        """Test that operation methods follow consistent naming patterns"""
        import ast

        identity_file = Path(__file__).parent.parent / "integrations" / "enterprise" / "identity_management.py"

        with open(identity_file, 'r') as f:
            source = f.read()

        tree = ast.parse(source)

        # Check that private operation methods start with underscore
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith('Integration'):
                for method in node.body:
                    if (
                        isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and method.name
                        not in ['__init__', 'initialize', 'cleanup', 'health_check', '_execute_operation_impl']
                        and not method.name.startswith('_')
                    ):
                        # Skip special methods and public interface methods
                        if not method.name.startswith('__'):
                            self.fail(f"Operation method {method.name} in {node.name} should be private (start with _)")


def run_syntax_tests():
    """Run all syntax and structure tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestEnterpriseIntegrationsSyntax, TestEnterpriseIntegrationLogic]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_syntax_tests()
    sys.exit(0 if success else 1)
