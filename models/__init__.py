"""
VLM (Vision-Language Model) abstraction package.

Provides a unified interface for all three supported models:
  - Ollama (Qwen2.5-VL-7B, LLaMA 3.2 Vision) — local inference via Docker
  - Gemini 2.0 Flash — Google cloud API (free up to 1500 req/day)

Usage:
    from models.registry import ModelRegistry
    registry = ModelRegistry.from_env()
    client = registry.get("qwen2.5vl:7b")
    response = client.generate(image, prompt)
"""