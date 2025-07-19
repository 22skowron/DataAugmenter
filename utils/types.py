from typing import List, TypedDict, Literal
from enum import Enum


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PromptType(ExplicitEnum):
    """
    Enum for supported prompt formats used in paraphrasing task.
    """

    BASIC = "basic"
    STRICT = "strict"
    FEW_SHOT = "few_shot"
    CUSTOM = "custom"

PromptTypeLiteral = Literal["basic", "strict", "few_shot", "custom"]
"""Supported prompt formats used in paraphrasing task."""


class ChatMessage(TypedDict):
    """
    Single message in a chat template, with sender role and content.
    """

    role: Literal["system", "user", "assistant"]
    content: str

ChatTemplate = List[ChatMessage]
"""A list of ChatMessage objects forming a prompt for chat-based models."""


class InputJSON(TypedDict):
    """
    Input JSON object ready for augmentation.
    """

    text: str
    text_hash: str
    text_source: str


class ParaphrasedJSON(InputJSON):
    """
    InputJSON extended with generated paraphrases.
    """

    paraphrases: list[str]


class Translation(TypedDict):
    """
    Forward and backward translations of a text segment.
    """

    forward_translation: str
    backward_translation: str


class TranslatedJSON(InputJSON):
    """
    InputJSON extended with for- and backward translations.
    """

    translations: list[Translation]