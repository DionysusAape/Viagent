"""LLM client for calling language models"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek



@dataclass
class ModelConfig:
    """Configuration for a model provider"""
    model_class: Type[BaseChatModel]
    env_key: Optional[str] = None
    base_url: Optional[str] = None
    requires_api_key: bool = True


class Provider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ALIBABA = "alibaba"
    KIMI = "kimi"
    AIHUBMIX = "aihubmix"
    YIZHAN = "yizhan"

    @property
    def config(self) -> ModelConfig:
        """Get the configuration for this provider"""
        PROVIDER_CONFIGS = {
            Provider.OPENAI: ModelConfig(
                model_class=ChatOpenAI,
                env_key="OPENAI_API_KEY",
            ),
            Provider.DEEPSEEK: ModelConfig(
                model_class=ChatDeepSeek,
                base_url="https://api.deepseek.com/v1",
                env_key="DEEPSEEK_API_KEY",
            ),
            Provider.ALIBABA: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                env_key="QWEN_API_KEY",
            ),
            Provider.KIMI: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://api.moonshot.cn/v1",
                env_key="KIMI_API_KEY",
            ),
            Provider.AIHUBMIX: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://api.aihubmix.com/v1",
                env_key="AIHUBMIX_API_KEY",
            ),
            Provider.YIZHAN: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://hk.yi-zhan.top/v1",
                env_key="YIZHAN_API_KEY",
            ),
        }
        return PROVIDER_CONFIGS[self]
