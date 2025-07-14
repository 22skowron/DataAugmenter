import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Iterator, Tuple, Literal, Optional, TypedDict

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding
)
import torch

from utils.types import (
    InputJSON,
    ParaphrasedJSON,
    TranslatedJSON,
    AugmentationStrategy,
    AugmentationStrategyLiteral,
    PromptType,
    PromptTypeLiteral
)
from utils.logger import logger


class DataAugmenter(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the full augmentation pipeline on a given text."""
        pass

    @staticmethod
    def save_augmentations():
        """Save generated augmentations to a file."""
        ...


class LLMDataAugmenter(DataAugmenter):
    def __init__(
            self,
            device: Optional[torch.device] = None
    ):
        self.device = self._init_device(device)

    @staticmethod
    def _init_device(device):
        if isinstance(device, torch.device):
            return device
        elif device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError("Invalid value for `device` parameter.")

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate augmented variations of the input text."""
        ...


class SpecificAugmenter(LLMDataAugmenter):
    def __init__(
            self,
            pretrained_model_name_or_path: Path | str,
            device: Optional[torch.device] = None,
            **kwargs
    ):
        super().__init__(device)

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device_map=self.device,
            torch_dtype="auto",
            **kwargs
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            padding_side='left',
            **kwargs
        )

    def run(self):
        pass

    def generate(self, text: str):
        pass


specific_augmenter = SpecificAugmenter()
specific_augmenter.generate("abc")
