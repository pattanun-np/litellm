from typing import Type, Union

from .chat.transformation import AnthropicConfig

__all__ = ["AnthropicConfig"]


def get_anthropic_config(
    url_route: str,
) -> Type[AnthropicConfig]:
    return AnthropicConfig
