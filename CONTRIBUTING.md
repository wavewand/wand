# Contributing to Wand

Thank you for your interest in contributing to Wand! We welcome contributions from the community.

## How to Contribute

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** and ensure tests pass
3. **Write or update tests** for your changes
4. **Ensure your code follows** the existing style
5. **Create a Pull Request** with a clear description

## Development Setup

```bash
git clone https://github.com/wavewand/wand.git
cd wand
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Testing

Run tests before submitting:
```bash
python -m pytest tests/
```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Keep functions focused and small
- Document complex logic

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
