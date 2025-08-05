"""
Smoke test for CI/CD pipeline
Tests basic functionality without optional dependencies
"""

import importlib
import sys

import pytest


class TestCISmoke:
    """Basic smoke tests that should always pass in CI"""

    def test_python_version(self):
        """Verify Python version is 3.10+"""
        assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"

    def test_core_imports(self):
        """Test that core modules can be imported"""
        modules = ['mcp', 'fastmcp', 'fastapi', 'pydantic', 'grpc', 'aiohttp', 'redis', 'sqlalchemy']

        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError as e:
                pytest.fail(f"Failed to import required module {module}: {e}")

    def test_wand_imports(self):
        """Test that main Wand modules can be imported"""
        # These should work with base requirements
        from config import get_config
        from tools.execution.base import ExecutionBackend

        assert get_config is not None
        assert ExecutionBackend is not None

    def test_optional_imports(self):
        """Test optional imports are handled gracefully"""
        try:
            import cv2

            has_opencv = True
        except ImportError:
            has_opencv = False

        try:
            import whisper

            has_whisper = True
        except ImportError:
            has_whisper = False

        # These are optional, so we just log their status
        print(f"OpenCV available: {has_opencv}")
        print(f"Whisper available: {has_whisper}")

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test that async code works"""
        import asyncio

        async def sample_async():
            await asyncio.sleep(0.001)
            return "success"

        result = await sample_async()
        assert result == "success"

    def test_config_defaults(self):
        """Test that default config can be created"""
        from config import Settings

        settings = Settings()
        assert settings is not None
        # Basic check that settings has expected attributes
        assert hasattr(settings, 'frameworks')
