"""
🔒 Security & Compliance Integrations

Security scanning, identity management, and compliance tools for Wand
"""

from .compliance import SnykIntegration, SonarQubeIntegration, VeracodeIntegration
from .security_tools import Auth0Integration, OktaIntegration, OnePasswordIntegration, VaultIntegration

# Initialize integration instances
vault_integration = VaultIntegration()
onepassword_integration = OnePasswordIntegration()
okta_integration = OktaIntegration()
auth0_integration = Auth0Integration()

veracode_integration = VeracodeIntegration()
snyk_integration = SnykIntegration()
sonarqube_integration = SonarQubeIntegration()

__all__ = [
    # Security tools
    "vault_integration",
    "onepassword_integration",
    "okta_integration",
    "auth0_integration",
    # Compliance & scanning
    "veracode_integration",
    "snyk_integration",
    "sonarqube_integration",
]
