"""
VLM (Vision-Language Model) abstraction package.

Provides a unified interface for the supported OpenRouter-backed models.

Usage:
    from models.registry import ModelRegistry
    registry = ModelRegistry.from_env()
    client = registry.get("openai/gpt-4o-mini")
    response = client.generate(image, prompt)
"""
