"""
🤖 AI/ML Platform Integrations

Advanced AI and machine learning capabilities for Wand
"""

from .core_platforms import (
    AnthropicIntegration,
    CohereIntegration,
    HuggingFaceIntegration,
    OpenAIIntegration,
    ReplicateIntegration,
)
from .ollama import OllamaIntegration
from .specialized_ai import DeepLIntegration, StabilityIntegration

# Initialize integration instances
huggingface_integration = HuggingFaceIntegration()
openai_integration = OpenAIIntegration()
anthropic_integration = AnthropicIntegration()
cohere_integration = CohereIntegration()
replicate_integration = ReplicateIntegration()

stability_integration = StabilityIntegration()
deepl_integration = DeepLIntegration()

# Configure Ollama with no rate limits (remote server)
ollama_config = {"rate_limit": {"enabled": False}}  # Disable rate limiting for remote Ollama server
ollama_integration = OllamaIntegration(ollama_config)

__all__ = [
    # Core AI platforms
    "huggingface_integration",
    "openai_integration",
    "anthropic_integration",
    "cohere_integration",
    "replicate_integration",
    # Specialized AI
    "stability_integration",
    "deepl_integration",
    # Local AI
    "ollama_integration",
]
