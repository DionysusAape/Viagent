"""LLM inference functions"""
import os
from typing import Dict, Any, Optional, List, Type, TypeVar
from dataclasses import dataclass
from pydantic import BaseModel
from util.logger import logger
from langchain_core.messages import HumanMessage
from .client import Provider, ModelConfig

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMConfig:
    """Configuration for LLM inference"""
    provider: str
    model: str
    temperature: float = 0.5
    max_retries: int = 3


def get_llm(config: LLMConfig) -> ModelConfig:
    """Get an LLM instance based on configuration"""
    provider = Provider(config.provider)
    model_config = provider.config

    if model_config.requires_api_key:
        api_key = os.getenv(model_config.env_key)
        if not api_key:
            raise ValueError(f"API key not found. Set {model_config.env_key} in your .env file.")

    model_kwargs = {
        "model": config.model,
        **({"temperature": config.temperature} if config.temperature else {}),
        **({"max_retries": config.max_retries} if config.max_retries else {}),
        **({"base_url": model_config.base_url} if model_config.base_url else {}),
        **({"api_key": api_key} if api_key else {}),
    }

    return model_config.model_class(**model_kwargs)

def call_llm(prompt: str, config: Dict[str, Any], pydantic_model: Type[T], images: Optional[List[str]] = None) -> T:
    """
    Make a call to the LLM with the given prompt and retry if it fails.
    """
    llm_config_kwargs = {
        "provider": config.get("provider"),
        "model": config.get("model"),
        "temperature": config.get("temperature", 0.5),
        "max_retries": config.get("max_retries", 3),
    }
    llm_config = LLMConfig(**llm_config_kwargs)
    llm = get_llm(llm_config)

    llm = llm.with_structured_output(pydantic_model, method="function_calling")

    # Build messages: support text-only or multimodal (text + images)
    if images:
        # Multimodal input: text + images
        content = [{"type": "text", "text": prompt}]
        for img in images:
            if img:
                content.append({"type": "image_url", "image_url": {"url": img}})
        messages = [HumanMessage(content=content)]
    else:
        # Text-only input
        messages = [HumanMessage(content=prompt)]

    last_error = None
    for attempt in range(llm_config.max_retries):
        try:
            response = llm.invoke(messages)
            if not response:
                raise ValueError("No response from LLM")
            return response
        except Exception as e:
            last_error = e
            error_msg = str(e)
            error_type = type(e).__name__
            logger.warning(f"Attempt {attempt + 1} of {llm_config.max_retries} failed to get response from LLM: {error_type}: {error_msg}")
            if attempt == llm_config.max_retries - 1:
                logger.error(f"Failed to get response from LLM after {llm_config.max_retries} attempts. {error_type}: {error_msg}")
                raise RuntimeError(f"LLM API call failed after {llm_config.max_retries} attempts. {error_type}: {error_msg}") from last_error
