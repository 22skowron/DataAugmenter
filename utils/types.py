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







class InputJSON(TypedDict):
    text: str
    text_hash: str
    text_source: str

class Paraphrase(TypedDict):
    paraphrase: str

class OutputJSON(InputJSON):
    paraphrases: List[Paraphrase]