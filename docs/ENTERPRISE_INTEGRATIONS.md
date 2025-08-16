# Enterprise Integrations for Wand

This document describes how to configure and use the enterprise identity management and communication integrations in Wand.

## Overview

Wand now includes comprehensive enterprise integrations for:

- **ServiceNow** - IT Service Management (incidents, users, records)
- **SailPoint** - Identity Security Cloud (identities, access requests, campaigns)
- **Microsoft Entra** - Azure AD identity management (users, groups, roles)
- **Britive** - Privileged access management (profiles, access requests, secrets)
- **Microsoft Teams** - Communication via webhooks (messages, cards, notifications)

## Installation

Install the enterprise dependencies:

```bash
pip install wand[enterprise]
```

Or install individual packages:

```bash
pip install pysnc msal msgraph-sdk azure-identity britive pymsteams
```

## Configuration

### ServiceNow

Set environment variables:

```bash
export SERVICENOW_INSTANCE_URL="https://your-instance.service-now.com"
export SERVICENOW_USERNAME="your-username"
export SERVICENOW_PASSWORD="your-password"
```

### SailPoint IdentityNow

Set environment variables:

```bash
export SAILPOINT_BASE_URL="https://your-tenant.api.identitynow.com"
export SAILPOINT_CLIENT_ID="your-client-id"
export SAILPOINT_CLIENT_SECRET="your-client-secret"
```

### Microsoft Entra (Azure AD)

Create an Azure AD app registration and set:

```bash
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
```

Required permissions:
- `User.Read.All`
- `Group.Read.All`
- `Directory.Read.All`
- `RoleManagement.ReadWrite.Directory` (for role assignments)

### Britive

Set environment variables:

```bash
export BRITIVE_TENANT="your-tenant-name"
export BRITIVE_API_TOKEN="your-api-token"
```

### Microsoft Teams

Configure webhook URLs for channels:

```bash
export TEAMS_WEBHOOK_URL="https://your-org.webhook.office.com/webhookb2/..."  # Default channel
export TEAMS_WEBHOOK_GENERAL="https://your-org.webhook.office.com/webhookb2/..."  # General channel
export TEAMS_WEBHOOK_ALERTS="https://your-org.webhook.office.com/webhookb2/..."  # Alerts channel
```

To get webhook URLs:
1. Go to your Teams channel
2. Click the "..." menu → "Connectors"
3. Search for "Incoming Webhook"
4. Configure and copy the webhook URL

## Usage Examples

### ServiceNow Operations

```python
# Create an incident
await servicenow(
    operation="create_incident",
    short_description="Server outage in production",
    description="Database server db01 is not responding",
    priority="1",
    assigned_to="john.doe"
)

# Query incidents
await servicenow(
    operation="query_records",
    table="incident",
    limit=10,
    query_filter="state=1^priority=1"
)

# Get user information
await servicenow(
    operation="get_user",
    username="john.doe"
)
```

### SailPoint Operations

```python
# Get identities
await sailpoint(
    operation="get_identities",
    limit=20,
    filters="displayName co 'John'"
)

# Request access for a user
await sailpoint(
    operation="request_access",
    identity_id="user-123",
    access_profile_ids=["profile-456", "profile-789"],
    justification="Quarterly access review"
)

# Launch certification campaign
await sailpoint(
    operation="launch_campaign",
    campaign_name="Q1 2024 Access Review",
    description="Quarterly review of user access",
    identities=["identity-1", "identity-2"]
)
```

### Microsoft Entra Operations

```python
# Get users
await entra(
    operation="get_users",
    limit=50,
    filter="startswith(displayName,'John')"
)

# Create a new user
await entra(
    operation="create_user",
    userPrincipalName="jane.doe@company.com",
    displayName="Jane Doe",
    passwordProfile={
        "password": "TempPass123!",
        "forceChangePasswordNextSignIn": True
    },
    givenName="Jane",
    surname="Doe",
    jobTitle="Software Engineer"
)

# Add user to group
await entra(
    operation="add_user_to_group",
    user_id="user-guid",
    group_id="group-guid"
)
```

### Britive Operations

```python
# List available profiles
await britive(
    operation="list_profiles",
    limit=20
)

# Request access to a profile
await britive(
    operation="request_access",
    profile_id="profile-123",
    justification="Emergency database access for incident resolution",
    duration=60  # minutes
)

# Check my pending requests
await britive(
    operation="get_my_requests",
    status="pending"
)
```

### Microsoft Teams Operations

```python
# Send simple message
await teams(
    operation="send_message",
    message="Hello from Wand! 👋",
    channel="general"
)

# Send notification with status
await teams(
    operation="send_notification",
    title="System Alert",
    message="High CPU usage detected on server prod-01",
    status="warning",
    details=[
        {"name": "Server", "value": "prod-01"},
        {"name": "CPU Usage", "value": "85%"},
        {"name": "Threshold", "value": "80%"}
    ],
    channel="alerts"
)

# Send custom card
await teams(
    operation="send_card",
    title="Deployment Complete",
    summary="Application deployment finished",
    theme_color="28a745",
    sections=[
        {
            "text": "Application v2.1.0 has been successfully deployed to production.",
            "facts": [
                {"name": "Version", "value": "v2.1.0"},
                {"name": "Environment", "value": "Production"},
                {"name": "Duration", "value": "5 minutes"}
            ]
        }
    ]
)
```

## Error Handling

All integrations include comprehensive error handling with:

- **Authentication errors** - Invalid credentials or expired tokens
- **Rate limiting** - Automatic retry with backoff
- **Network errors** - Connection timeouts and retries
- **Validation errors** - Missing required parameters

Example error response:
```json
{
    "success": false,
    "error": "Authentication failed: Invalid credentials",
    "integration": "servicenow",
    "operation": "create_incident",
    "correlation_id": "uuid-123",
    "troubleshooting": "Verify API credentials and permissions"
}
```

## Health Monitoring

Check integration health:

```python
# Each integration provides health check
integration = ServiceNowIntegration()
await integration.initialize()
health = await integration.health_check()
print(health)
```

Example health response:
```json
{
    "status": "healthy",
    "enabled": true,
    "connection": "active",
    "instance_url": "https://dev12345.service-now.com",
    "metrics": {
        "requests_total": 150,
        "success_rate": 0.95,
        "average_response_time": 0.8
    }
}
```

## Security Best Practices

1. **Use environment variables** for sensitive configuration
2. **Rotate API tokens** regularly
3. **Use least privilege** - only grant necessary permissions
4. **Monitor access logs** in your enterprise systems
5. **Enable audit logging** in Wand for compliance

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify credentials in environment variables
   - Check API token expiration
   - Confirm required permissions are granted

2. **Network Connectivity**
   - Test network connectivity to service endpoints
   - Check firewall rules and proxy settings
   - Verify SSL/TLS certificate trust

3. **Rate Limiting**
   - Review service-specific rate limits
   - Implement appropriate delays between requests
   - Consider upgrading service plans if needed

4. **Permission Errors**
   - Review required permissions for each operation
   - Ensure service accounts have necessary roles
   - Check organization-level restrictions

### Debug Mode

Enable verbose logging:

```bash
export WAND_LOG_LEVEL=DEBUG
```

### Support

For enterprise integration support:
- Check service-specific documentation
- Review audit logs in your enterprise systems
- Contact your enterprise administrators for permission issues

## Roadmap

Planned enterprise integrations:
- Okta identity management
- Auth0 identity platform
- CyberArk privileged access
- Ping Identity
- Active Directory on-premises
