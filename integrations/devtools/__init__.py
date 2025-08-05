"""
🛠 Developer Tools & Infrastructure Integrations

Container orchestration, monitoring, and testing tools for Wand
"""

from .containers import DockerIntegration, KubernetesIntegration, TerraformIntegration
from .monitoring import DatadogIntegration, PrometheusIntegration, SentryIntegration
from .testing import PlaywrightIntegration, PostmanIntegration, SeleniumIntegration

# Initialize integration instances
docker_integration = DockerIntegration()
kubernetes_integration = KubernetesIntegration()
terraform_integration = TerraformIntegration()

prometheus_integration = PrometheusIntegration()
datadog_integration = DatadogIntegration()
sentry_integration = SentryIntegration()

selenium_integration = SeleniumIntegration()
playwright_integration = PlaywrightIntegration()
postman_integration = PostmanIntegration()

__all__ = [
    # Container & orchestration
    "docker_integration",
    "kubernetes_integration",
    "terraform_integration",
    # Monitoring & observability
    "prometheus_integration",
    "datadog_integration",
    "sentry_integration",
    # Testing & quality
    "selenium_integration",
    "playwright_integration",
    "postman_integration",
]
