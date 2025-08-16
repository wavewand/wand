#!/usr/bin/env python3
"""
Basic tests for enterprise integrations that work without optional dependencies.
These tests focus on structure validation and basic functionality.
"""

import sys
import unittest
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class TestEnterpriseIntegrationStructure(unittest.TestCase):
    """Test enterprise integration structure and basic functionality"""

    def test_servicenow_integration_structure(self):
        """Test ServiceNow integration class structure"""
        try:
            from integrations.enterprise.identity_management import ServiceNowIntegration

            # Test basic structure without requiring external dependencies
            config = {"instance_url": "https://test.service-now.com", "username": "test_user", "password": "test_pass"}

            integration = ServiceNowIntegration(config)
            self.assertEqual(integration.name, "servicenow")
            self.assertIn("instance_url", integration.config)
            self.assertEqual(integration.REQUIRED_CONFIG_KEYS, ["instance_url", "username", "password"])
        except ImportError as e:
            self.skipTest(f"ServiceNow dependencies not available: {e}")

    def test_sailpoint_integration_structure(self):
        """Test SailPoint integration class structure"""
        try:
            from integrations.enterprise.identity_management import SailPointIntegration

            config = {
                "base_url": "https://test.api.identitynow.com",
                "client_id": "test_client",
                "client_secret": "test_secret",
            }

            integration = SailPointIntegration(config)
            self.assertEqual(integration.name, "sailpoint")
            self.assertIn("base_url", integration.config)
            self.assertEqual(integration.REQUIRED_CONFIG_KEYS, ["base_url", "client_id", "client_secret"])
        except ImportError as e:
            self.skipTest(f"SailPoint dependencies not available: {e}")

    def test_microsoft_entra_integration_structure(self):
        """Test Microsoft Entra integration class structure"""
        try:
            from integrations.enterprise.identity_management import MicrosoftEntraIntegration

            config = {"tenant_id": "test_tenant", "client_id": "test_client", "client_secret": "test_secret"}

            integration = MicrosoftEntraIntegration(config)
            self.assertEqual(integration.name, "microsoft_entra")
            self.assertIn("tenant_id", integration.config)
            self.assertEqual(integration.REQUIRED_CONFIG_KEYS, ["tenant_id", "client_id", "client_secret"])
        except ImportError as e:
            self.skipTest(f"Microsoft Entra dependencies not available: {e}")

    def test_britive_integration_structure(self):
        """Test Britive integration class structure"""
        try:
            from integrations.enterprise.identity_management import BritiveIntegration

            config = {"tenant": "test-tenant", "api_token": "test_token"}

            integration = BritiveIntegration(config)
            self.assertEqual(integration.name, "britive")
            self.assertIn("tenant", integration.config)
            self.assertEqual(integration.REQUIRED_CONFIG_KEYS, ["tenant", "api_token"])
        except ImportError as e:
            self.skipTest(f"Britive dependencies not available: {e}")

    def test_microsoft_teams_integration_structure(self):
        """Test Microsoft Teams integration class structure"""
        try:
            from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

            config = {"webhooks": {"default": "https://company.webhook.office.com/webhookb2/test"}}

            integration = MicrosoftTeamsIntegration(config)
            self.assertEqual(integration.name, "microsoft_teams")
            self.assertIn("webhooks", integration.config)
        except ImportError as e:
            self.skipTest(f"Microsoft Teams dependencies not available: {e}")

    def test_teams_message_builder(self):
        """Test Teams message builder helper class"""
        try:
            from integrations.productivity.teams_communication import TeamsMessageBuilder

            builder = TeamsMessageBuilder()
            card = (
                builder.set_title("Test Card")
                .set_theme_color("28a745")
                .add_section(text="This is a test")
                .add_fact("Status", "Success")
                .add_action("View Details", "https://example.com")
                .build()
            )

            self.assertEqual(card["title"], "Test Card")
            self.assertEqual(card["themeColor"], "28a745")
            self.assertEqual(len(card["sections"]), 1)
            self.assertEqual(card["sections"][0]["facts"][0]["name"], "Status")
            self.assertEqual(len(card["potentialAction"]), 1)
        except ImportError as e:
            self.skipTest(f"Teams message builder dependencies not available: {e}")

    def test_all_integrations_inherit_base(self):
        """Test that all enterprise integrations inherit from BaseIntegration"""
        try:
            from integrations.base.integration_base import BaseIntegration
            from integrations.enterprise.identity_management import (
                BritiveIntegration,
                MicrosoftEntraIntegration,
                SailPointIntegration,
                ServiceNowIntegration,
            )
            from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

            # Test inheritance
            self.assertTrue(issubclass(ServiceNowIntegration, BaseIntegration))
            self.assertTrue(issubclass(SailPointIntegration, BaseIntegration))
            self.assertTrue(issubclass(MicrosoftEntraIntegration, BaseIntegration))
            self.assertTrue(issubclass(BritiveIntegration, BaseIntegration))
            self.assertTrue(issubclass(MicrosoftTeamsIntegration, BaseIntegration))

            # Test that all have required methods
            for integration_class in [
                ServiceNowIntegration,
                SailPointIntegration,
                MicrosoftEntraIntegration,
                BritiveIntegration,
                MicrosoftTeamsIntegration,
            ]:
                self.assertTrue(hasattr(integration_class, 'initialize'))
                self.assertTrue(hasattr(integration_class, 'cleanup'))
                self.assertTrue(hasattr(integration_class, 'health_check'))
                self.assertTrue(hasattr(integration_class, '_execute_operation_impl'))

            # Test that identity management integrations have REQUIRED_CONFIG_KEYS
            for integration_class in [
                ServiceNowIntegration,
                SailPointIntegration,
                MicrosoftEntraIntegration,
                BritiveIntegration,
            ]:
                self.assertTrue(hasattr(integration_class, 'REQUIRED_CONFIG_KEYS'))

        except ImportError as e:
            self.skipTest(f"Enterprise integration dependencies not available: {e}")


class TestEnterpriseIntegrationConfig(unittest.TestCase):
    """Test enterprise integration configuration handling"""

    def test_servicenow_config_validation(self):
        """Test ServiceNow configuration validation"""
        try:
            from integrations.enterprise.identity_management import ServiceNowIntegration

            # Test missing config - should create integration but fail during initialization
            integration = ServiceNowIntegration({})
            self.assertEqual(integration.name, "servicenow")

            # Test complete config
            complete_config = {
                "instance_url": "https://test.service-now.com",
                "username": "test_user",
                "password": "test_pass",
            }
            integration = ServiceNowIntegration(complete_config)
            self.assertEqual(integration.name, "servicenow")
            self.assertEqual(integration.config["instance_url"], "https://test.service-now.com")

        except ImportError as e:
            self.skipTest(f"ServiceNow dependencies not available: {e}")

    def test_teams_config_validation(self):
        """Test Teams configuration validation"""
        try:
            from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

            # Test with valid webhook config
            config = {
                "webhooks": {
                    "default": "https://company.webhook.office.com/webhookb2/test",
                    "alerts": "https://company.webhook.office.com/webhookb2/alerts",
                }
            }

            integration = MicrosoftTeamsIntegration(config)
            self.assertIn("webhooks", integration.config)
            self.assertEqual(len(integration.config["webhooks"]), 2)

        except ImportError as e:
            self.skipTest(f"Teams dependencies not available: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
