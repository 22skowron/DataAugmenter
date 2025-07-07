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

class AugmentationStrategy(ExplicitEnum):
    PARAPHRASING = "paraphrasing"
    TRANSLATION = "translation"
AugmentationStrategyLiteral = Literal["paraphrasing", "translation"]

class PromptType(ExplicitEnum):
    BASIC = "basic"
    STRICT = "strict"
    FEW_SHOT = "few_shot"
    CUSTOM = "custom"
PromptTypeLiteral = Literal["basic", "strict", "few_shot", "custom"]

# JSON types
class InputJSON(TypedDict):
    text: str
    text_hash: str
    text_source: str

class ParaphrasedJSON(InputJSON):
    paraphrases: list[str]

class Translation(TypedDict):
    forward_translation: str
    backward_translation: str

class TranslatedJSON(InputJSON):
    translations: list[Translation]
