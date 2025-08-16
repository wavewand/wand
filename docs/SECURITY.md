# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **DO NOT** create a public GitHub issue
2. Email the security team at: security@wavewand.io (or create a private security advisory on GitHub)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: 1-2 weeks
  - High: 2-4 weeks
  - Medium: 4-8 weeks
  - Low: Next regular release

## Security Best Practices

When using Wand:

1. **Never commit secrets**: Use environment variables
2. **Keep dependencies updated**: Regular security updates
3. **Use secure defaults**: Enable authentication and rate limiting
4. **Validate inputs**: Especially for command execution
5. **Principle of least privilege**: Limit execution permissions

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help us keep Wand secure.
