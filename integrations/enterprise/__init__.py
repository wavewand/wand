"""
🏢 Enterprise & Business Integrations

CRM, project management, and business automation tools for Wand
"""

from .crm_sales import HubSpotIntegration, PipedriveIntegration, SalesforceIntegration, StripeIntegration
from .hr_operations import BambooHRIntegration, HarvestIntegration, TogglIntegration, WorkdayIntegration
from .project_mgmt import AsanaIntegration, JiraIntegration, LinearIntegration, MondayIntegration, TrelloIntegration

# Initialize integration instances
salesforce_integration = SalesforceIntegration()
hubspot_integration = HubSpotIntegration()
pipedrive_integration = PipedriveIntegration()
stripe_integration = StripeIntegration()

jira_integration = JiraIntegration()
asana_integration = AsanaIntegration()
trello_integration = TrelloIntegration()
linear_integration = LinearIntegration()
monday_integration = MondayIntegration()

workday_integration = WorkdayIntegration()
bamboohr_integration = BambooHRIntegration()
toggl_integration = TogglIntegration()
harvest_integration = HarvestIntegration()

__all__ = [
    # CRM & Sales
    "salesforce_integration",
    "hubspot_integration",
    "pipedrive_integration",
    "stripe_integration",
    # Project Management
    "jira_integration",
    "asana_integration",
    "trello_integration",
    "linear_integration",
    "monday_integration",
    # HR & Operations
    "workday_integration",
    "bamboohr_integration",
    "toggl_integration",
    "harvest_integration",
]
