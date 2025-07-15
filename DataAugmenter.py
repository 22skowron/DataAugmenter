import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Literal, Optional, TypedDict

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
    def save_augmentations(
            augmentations: list[dict],
            file: str | Path,
            mode: Literal["a", "w"] = "w",
            encoding: str | None = "utf-8"
    ):
        """
        Save generated augmentations to a file in JSON Lines (JSONL) format.

        Args:
            augmentations: A list of augmentation objects to save.
            file: The file path where augmentations will be saved.
            mode: File mode — 'a' to append or 'w' to overwrite. Defaults to 'w'.
            encoding: Encoding used to write the file. Defaults to 'utf-8'.
        """
        with open(file, mode, encoding=encoding) as f:
            for obj in augmentations:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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

    @abstractmethod
    def run(
            self,
            batch_size: int,
            num_augmentations: int,
            output_file: str | Path,
            data: Optional[list[InputJSON]] = None,
            input_file: Optional[str | Path] = None,
            start_line_idx: int = 1,
            len_factor: float = 1.4,
            temperature: float = 0.7,
            *args, **kwargs
    ):
        pass


class ParaphrasingAugmenter(LLMDataAugmenter):
    def __init__(
            self,
            pretrained_model_name_or_path: str | Path,
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

    def run(
            self,
            batch_size: int,
            num_augmentations: int,
            output_file: str | Path,
            data: Optional[list[InputJSON]] = None,
            input_file: Optional[str | Path] = None,
            start_line_idx: int = 1,
            len_factor: float = 1.4,
            temperature: float = 0.7,
            prompt_type: PromptTypeLiteral = "basic",
            **kwargs
    ):
        pass

    def generate(self, text: str):
        pass
