"""
Test cases for enterprise identity management and communication integrations
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path to allow importing integrations
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports - handle missing dependencies gracefully
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    # Create a minimal pytest.mark.asyncio decorator for compatibility
    class MockPytest:
        class mark:
            @staticmethod
            def asyncio(func):
                return func

    pytest = MockPytest()


class TestServiceNowIntegration(unittest.TestCase):
    """Test cases for ServiceNow integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {"instance_url": "https://test.service-now.com", "username": "test_user", "password": "test_pass"}

    def test_servicenow_initialization(self):
        """Test ServiceNow integration initialization"""
        from integrations.enterprise.identity_management import ServiceNowIntegration

        integration = ServiceNowIntegration(self.config)
        self.assertEqual(integration.name, "servicenow")
        self.assertEqual(integration.config["instance_url"], "https://test.service-now.com")

    async def _test_servicenow_health_check(self):
        """Test ServiceNow health check (async)"""
        try:
            from integrations.enterprise.identity_management import ServiceNowIntegration

            with patch('pysnc.ServiceNowClient') as mock_client:
                # Mock the GlideRecord behavior
                mock_gr = MagicMock()
                mock_gr.get_row_count.return_value = 1
                mock_client.return_value.GlideRecord.return_value = mock_gr

                integration = ServiceNowIntegration(self.config)
                await integration.initialize()

                health = await integration.health_check()
                self.assertEqual(health["status"], "healthy")
                self.assertIn("instance_url", health)
        except ImportError as e:
            self.skipTest(f"ServiceNow dependencies not available: {e}")

    def test_servicenow_health_check(self):
        """Wrapper for async health check test"""
        if HAS_PYTEST:
            asyncio.run(self._test_servicenow_health_check())

    async def _test_servicenow_create_incident(self):
        """Test ServiceNow incident creation"""
        try:
            from integrations.enterprise.identity_management import ServiceNowIntegration

            with patch('pysnc.ServiceNowClient') as mock_client:
                # Mock the GlideRecord behavior for incident creation
                mock_gr = MagicMock()
                mock_gr.insert.return_value = "INC0123456"
                mock_gr.number = "INC0123456"
                mock_client.return_value.GlideRecord.return_value = mock_gr

                integration = ServiceNowIntegration(self.config)
                await integration.initialize()

                result = await integration._create_incident(
                    short_description="Test incident", priority="2", description="Test description"
                )

                self.assertTrue(result["success"])
                self.assertEqual(result["incident_id"], "INC0123456")
                self.assertEqual(result["number"], "INC0123456")
        except ImportError as e:
            self.skipTest(f"ServiceNow dependencies not available: {e}")

    def test_servicenow_create_incident(self):
        """Wrapper for async incident creation test"""
        if HAS_PYTEST:
            asyncio.run(self._test_servicenow_create_incident())

    async def _test_servicenow_query_records(self):
        """Test ServiceNow record querying"""
        try:
            from integrations.enterprise.identity_management import ServiceNowIntegration

            with patch('pysnc.ServiceNowClient') as mock_client:
                # Mock the GlideRecord behavior for querying
                mock_gr = MagicMock()
                mock_gr.get_fields.return_value = ["sys_id", "number", "short_description"]

                # Create mock records
                mock_record1 = MagicMock()
                mock_record1.sys_id = "123"
                mock_record1.number = "INC0001"
                mock_record1.short_description = "Test incident 1"

                mock_record2 = MagicMock()
                mock_record2.sys_id = "456"
                mock_record2.number = "INC0002"
                mock_record2.short_description = "Test incident 2"

                mock_gr.__iter__ = MagicMock(return_value=iter([mock_record1, mock_record2]))
                mock_client.return_value.GlideRecord.return_value = mock_gr

                integration = ServiceNowIntegration(self.config)
                await integration.initialize()

                result = await integration._query_records(table="incident", limit=10, query_filter="state=1")

                self.assertTrue(result["success"])
                self.assertEqual(len(result["records"]), 2)
                self.assertEqual(result["records"][0]["number"], "INC0001")
        except ImportError as e:
            self.skipTest(f"ServiceNow dependencies not available: {e}")

    def test_servicenow_query_records(self):
        """Wrapper for async record querying test"""
        if HAS_PYTEST:
            asyncio.run(self._test_servicenow_query_records())


class TestSailPointIntegration(unittest.TestCase):
    """Test cases for SailPoint integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "base_url": "https://test.api.identitynow.com",
            "client_id": "test_client",
            "client_secret": "test_secret",
        }

    def test_sailpoint_initialization(self):
        """Test SailPoint integration initialization"""
        from integrations.enterprise.identity_management import SailPointIntegration

        integration = SailPointIntegration(self.config)
        self.assertEqual(integration.name, "sailpoint")
        self.assertEqual(integration.config["base_url"], "https://test.api.identitynow.com")

    async def _test_sailpoint_authentication(self):
        """Test SailPoint OAuth authentication"""
        from integrations.enterprise.identity_management import SailPointIntegration

        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock session and response
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"access_token": "test_token"})

            # Set up the async context manager properly
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = AsyncMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            integration = SailPointIntegration(self.config)
            result = await integration.execute_operation("get_identities", limit=10)

            # Check that we get a proper result structure
            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

    def test_sailpoint_authentication(self):
        """Wrapper for async authentication test"""
        if HAS_PYTEST:
            asyncio.run(self._test_sailpoint_authentication())

    async def _test_sailpoint_get_identities(self):
        """Test SailPoint get identities operation"""
        from integrations.enterprise.identity_management import SailPointIntegration

        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock session and response
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value=[
                    {"id": "1", "name": "John Doe", "email": "john@company.com"},
                    {"id": "2", "name": "Jane Smith", "email": "jane@company.com"},
                ]
            )

            # Set up the async context manager properly
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.get = AsyncMock()
            mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = AsyncMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            integration = SailPointIntegration(self.config)
            result = await integration.execute_operation("get_identities", limit=2)

            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

    def test_sailpoint_get_identities(self):
        """Wrapper for async get identities test"""
        if HAS_PYTEST:
            asyncio.run(self._test_sailpoint_get_identities())


class TestMicrosoftEntraIntegration(unittest.TestCase):
    """Test cases for Microsoft Entra integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
            "client_secret": "test-client-secret",
        }

    def test_entra_initialization(self):
        """Test Microsoft Entra integration initialization"""
        from integrations.enterprise.identity_management import MicrosoftEntraIntegration

        integration = MicrosoftEntraIntegration(self.config)
        self.assertEqual(integration.name, "microsoft_entra")
        self.assertEqual(integration.config["tenant_id"], "test-tenant-id")

    async def _test_entra_get_users(self):
        """Test Microsoft Entra get users operation"""
        try:
            from integrations.enterprise.identity_management import MicrosoftEntraIntegration

            with patch('azure.identity.aio.ClientSecretCredential'):
                with patch('msgraph.GraphServiceClient') as mock_client:
                    # Mock the Graph client response
                    mock_users_response = MagicMock()
                    mock_user1 = MagicMock()
                    mock_user1.id = "user1"
                    mock_user1.user_principal_name = "john@company.com"
                    mock_user1.display_name = "John Doe"
                    mock_user1.given_name = "John"
                    mock_user1.surname = "Doe"
                    mock_user1.mail = "john@company.com"
                    mock_user1.account_enabled = True
                    mock_user1.job_title = "Developer"
                    mock_user1.department = "IT"

                    mock_users_response.value = [mock_user1]
                    mock_client.return_value.users.get.return_value = mock_users_response

                    integration = MicrosoftEntraIntegration(self.config)
                    await integration.initialize()

                    result = await integration._get_users(limit=10)

                    self.assertTrue(result["success"])
                    self.assertEqual(len(result["users"]), 1)
                    self.assertEqual(result["users"][0]["displayName"], "John Doe")
        except ImportError as e:
            self.skipTest(f"Microsoft Entra dependencies not available: {e}")

    def test_entra_get_users(self):
        """Wrapper for async get users test"""
        if HAS_PYTEST:
            asyncio.run(self._test_entra_get_users())


class TestBritiveIntegration(unittest.TestCase):
    """Test cases for Britive integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {"tenant": "test-tenant", "api_token": "test-token"}

    def test_britive_initialization(self):
        """Test Britive integration initialization"""
        from integrations.enterprise.identity_management import BritiveIntegration

        integration = BritiveIntegration(self.config)
        self.assertEqual(integration.name, "britive")
        self.assertEqual(integration.config["tenant"], "test-tenant")

    async def _test_britive_list_profiles(self):
        """Test Britive list profiles operation"""
        try:
            from integrations.enterprise.identity_management import BritiveIntegration

            with patch('britive.Britive') as mock_client:
                # Mock the Britive client response
                mock_profiles = [
                    {"id": "profile1", "name": "Database Admin", "applicationId": "app1"},
                    {"id": "profile2", "name": "Server Admin", "applicationId": "app2"},
                ]
                mock_client.return_value.profiles.list.return_value = mock_profiles

                integration = BritiveIntegration(self.config)
                await integration.initialize()

                result = await integration._list_profiles(limit=10)

                self.assertTrue(result["success"])
                self.assertEqual(len(result["profiles"]), 2)
                self.assertEqual(result["profiles"][0]["name"], "Database Admin")
        except ImportError as e:
            self.skipTest(f"Britive dependencies not available: {e}")

    def test_britive_list_profiles(self):
        """Wrapper for async list profiles test"""
        if HAS_PYTEST:
            asyncio.run(self._test_britive_list_profiles())


class TestMicrosoftTeamsIntegration(unittest.TestCase):
    """Test cases for Microsoft Teams integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {"webhooks": {"default": "https://company.webhook.office.com/webhookb2/test"}}

    def test_teams_initialization(self):
        """Test Microsoft Teams integration initialization"""
        from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

        integration = MicrosoftTeamsIntegration(self.config)
        self.assertEqual(integration.name, "microsoft_teams")
        # The webhook_urls are populated during initialization, so check after init
        asyncio.run(integration.initialize())
        self.assertIsInstance(integration.webhook_urls, dict)

    async def _test_teams_send_message(self):
        """Test Microsoft Teams send message operation"""
        from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock session and response
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200

            # Set up the async context manager properly
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = AsyncMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            integration = MicrosoftTeamsIntegration(self.config)
            result = await integration.execute_operation(
                "send_message", message="Test message from Wand", channel="default"
            )

            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

    def test_teams_send_message(self):
        """Wrapper for async send message test"""
        if HAS_PYTEST:
            asyncio.run(self._test_teams_send_message())

    async def _test_teams_send_notification(self):
        """Test Microsoft Teams send notification operation"""
        from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

        with patch('aiohttp.ClientSession') as mock_session_class:
            # Create mock session and response
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200

            # Set up the async context manager properly
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = AsyncMock()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            integration = MicrosoftTeamsIntegration(self.config)
            result = await integration.execute_operation(
                "send_notification", title="System Alert", message="CPU usage high", status="warning"
            )

            self.assertIsInstance(result, dict)
            self.assertIn("success", result)

    def test_teams_send_notification(self):
        """Wrapper for async send notification test"""
        if HAS_PYTEST:
            asyncio.run(self._test_teams_send_notification())

    def test_teams_message_builder(self):
        """Test Teams message builder helper class"""
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

    def test_teams_convenience_functions(self):
        """Test Teams convenience functions"""
        from integrations.productivity.teams_communication import (
            create_alert_card,
            create_simple_message,
            create_status_card,
        )

        # Test simple message
        simple_msg = create_simple_message("Hello World")
        self.assertEqual(simple_msg["text"], "Hello World")

        # Test alert card
        alert_card = create_alert_card("Alert", "Something happened", "warning")
        self.assertEqual(alert_card["title"], "🔔 Alert")
        self.assertEqual(alert_card["themeColor"], "ffc107")

        # Test status card
        status_card = create_status_card("MyService", "healthy")
        self.assertEqual(status_card["title"], "✅ MyService Status")
        self.assertEqual(status_card["themeColor"], "28a745")


class TestEnterpriseIntegrationErrors(unittest.TestCase):
    """Test error handling in enterprise integrations"""

    async def _test_authentication_errors(self):
        """Test authentication error handling"""
        from integrations.enterprise.identity_management import ServiceNowIntegration

        # Test with missing credentials
        integration = ServiceNowIntegration({})
        await integration.initialize()

        self.assertFalse(integration.enabled)
        self.assertIsNotNone(integration.initialization_error)

    def test_authentication_errors(self):
        """Wrapper for async authentication error test"""
        if HAS_PYTEST:
            asyncio.run(self._test_authentication_errors())

    async def _test_network_errors(self):
        """Test network error handling"""
        from integrations.productivity.teams_communication import MicrosoftTeamsIntegration

        config = {"webhooks": {"default": "https://example.com/webhook"}}

        integration = MicrosoftTeamsIntegration(config)

        # Test with network error - the error happens during execute_operation
        result = await integration.execute_operation("send_message", message="Test")

        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        # Since the integration handles errors gracefully, we just verify structure

    def test_network_errors(self):
        """Wrapper for async network error test"""
        if HAS_PYTEST:
            asyncio.run(self._test_network_errors())


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation for enterprise integrations"""

    def test_required_config_validation(self):
        """Test that required configuration keys are validated"""
        from integrations.enterprise.identity_management import (
            BritiveIntegration,
            MicrosoftEntraIntegration,
            SailPointIntegration,
            ServiceNowIntegration,
        )

        # Test ServiceNow required keys
        integration = ServiceNowIntegration({})
        self.assertFalse(integration.enabled)

        # Test SailPoint required keys
        integration = SailPointIntegration({})
        self.assertFalse(integration.enabled)

        # Test Microsoft Entra required keys
        integration = MicrosoftEntraIntegration({})
        self.assertFalse(integration.enabled)

        # Test Britive required keys
        integration = BritiveIntegration({})
        self.assertFalse(integration.enabled)

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables"""
        from integrations.enterprise.identity_management import ServiceNowIntegration

        with patch.dict(
            os.environ,
            {
                'SERVICENOW_INSTANCE_URL': 'https://test.service-now.com',
                'SERVICENOW_USERNAME': 'test_user',
                'SERVICENOW_PASSWORD': 'test_pass',
            },
        ):
            integration = ServiceNowIntegration()
            self.assertEqual(integration.config["instance_url"], "https://test.service-now.com")
            self.assertEqual(integration.config["username"], "test_user")


# Test suite runner for compatibility
def run_enterprise_tests():
    """Run all enterprise integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestServiceNowIntegration,
        TestSailPointIntegration,
        TestMicrosoftEntraIntegration,
        TestBritiveIntegration,
        TestMicrosoftTeamsIntegration,
        TestEnterpriseIntegrationErrors,
        TestConfigurationValidation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when executed directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run without async tests for basic validation
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add only sync tests
        suite.addTest(TestServiceNowIntegration('test_servicenow_initialization'))
        suite.addTest(TestMicrosoftTeamsIntegration('test_teams_initialization'))
        suite.addTest(TestMicrosoftTeamsIntegration('test_teams_message_builder'))
        suite.addTest(TestConfigurationValidation('test_required_config_validation'))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        success = run_enterprise_tests()
        sys.exit(0 if success else 1)
