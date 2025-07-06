from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from typing import List, Iterator, Tuple, Literal, Optional, TypedDict

from linear_script import outputs, max_new_tokens
from utils.types import (
    InputJSON,
    Paraphrase,
    OutputJSON,
    AugmentationStrategy,
    AugmentationStrategyLiteral,
    PromptType,
    PromptTypeLiteral
)
from utils.logger import logger
from pathlib import Path
import json
import torch
import transformers
import time


class DataAugmenter:
    def __init__(
            self,
            paraphrasing_model: Optional[Path | str] = None,
            translation_model: Optional[Path | str] = None,
            device: Optional[torch.device] = None
    ):
        self.paraphrasing_model = None
        self.paraphrasing_tokenizer = None
        self.translation_model = None
        self.translation_tokenizer = None

        if isinstance(device, torch.device):
            self.device = device
        elif device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError("Invalid value for `device` parameter.")

        if paraphrasing_model:
            self.initialize_paraphrasing_model(paraphrasing_model)
        if translation_model:
            self.initialize_translation_model(translation_model)

    def initialize_paraphrasing_model(self, paraphrasing_model: Path | str):
        self.paraphrasing_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=paraphrasing_model,
            device_map=self.device,
            torch_dtype="auto"
        )
        self.paraphrasing_tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=paraphrasing_model,
            padding_side='left'
        )

    def initialize_translation_model(self, translation_model: Path | str):
        self.translation_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=translation_model,
            device_map="auto",
            torch_dtype="auto"
        )
        self.translation_tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=translation_model
        )

    def load_input(self):
        ...

    def batch_load_input(self):
        ...

    def save_augmentations(self):
        ...

    def get_prompt(
            self,
            og_text: str,
            prompt_type: PromptTypeLiteral = "basic",
            paraphrase_examples: Optional[list[str]] = None,
            custom_prompt: Optional[list[dict[str, str]]] = None
    ) -> list[dict[str, str]]:
        prompt_type = PromptType(prompt_type)

        messages = []
        if prompt_type == PromptType.BASIC:
            messages = [
                {"role": "system", "content": "Jesteś asystentem AI, który specjalizuje się w parafrazowaniu tekstu."},
                {"role": "user", "content": f"Proszę sparafrazuj następujący tekst: {og_text}"}
            ]
        elif prompt_type == PromptType.STRICT:
            raise NotImplementedError
        elif prompt_type == PromptType.FEW_SHOT:
            raise NotImplementedError
        elif prompt_type == PromptType.CUSTOM:
            raise NotImplementedError

        return messages

    def get_prompt_str(
            self,
            og_text: str,
            prompt_type: PromptTypeLiteral = "basic",
            paraphrase_examples: Optional[list[str]] = None,
            custom_prompt: Optional[list[dict[str, str]]] = None
    ) -> str:
        if not self.paraphrasing_tokenizer:
            raise ValueError(
                "Tokenizer of a paraphrasing LLM is not set. Please initialize it before calling get_prompt_str.")

        prompt = self.get_prompt(
            og_text=og_text,
            prompt_type=prompt_type,
            paraphrase_examples=paraphrase_examples,
            custom_prompt=custom_prompt
        )

        return self.paraphrasing_tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

    def tokenize_paraphrases(
            self,
            prompts: list[str],
            padding: str | bool = 'longest',
            truncation: str | bool = False,
            **kwargs
    ) -> transformers.BatchEncoding:
        if not self.paraphrasing_tokenizer:
            raise ValueError(
                "Tokenizer of a paraphrasing LLM is not set. Please initialize it before calling tokenize_paraphrases.")

        return self.paraphrasing_tokenizer(
            prompts,
            return_tensors='pt',
            padding=padding,
            padding_side='left',
            truncation=truncation,
            **kwargs
        ).to(self.device)

    def tokenize_translations(self) -> transformers.BatchEncoding:
        ...

    def get_sensible_max_new_tokens(
            self,
            num_text_tokens: int,
            augmentation_strategy: AugmentationStrategyLiteral,
            num_prompt_template_tokens: Optional[int] = None,
            len_factor: float = 1.4,
    ) -> int:
        augmentation_strategy = AugmentationStrategy(augmentation_strategy)

        max_new_tokens: int
        if augmentation_strategy == AugmentationStrategy.PARAPHRASING:
            if not self.paraphrasing_tokenizer:
                raise ValueError(
                    "Tokenizer of a paraphrasing LLM is not set. Please initialize it before calling tokenize_paraphrases.")
            if not isinstance(num_prompt_template_tokens, int):
                raise ValueError(
                    f"Invalid value for `num_prompt_template_tokens`. "
                    f"When augmentation_strategy is set to {AugmentationStrategy.PARAPHRASING.value} `num_prompt_template_tokens` is mandatory with type int."
                )

            ...

    def generate_paraphrases(
            self,
            inputs: transformers.BatchEncoding | torch.Tensor,
            num_paraphrases: int,
            num_input_tokens: int,
            max_new_tokens: int,
            temperature: int = 0.7,
            model: Optional[str] = None,
            **kwargs
    ) -> list[str]:
        if self.paraphrasing_model is None or self.paraphrasing_tokenizer is None:
            raise ValueError(
                "Paraphrasing LLM is not set. Please initialize it before calling generate_paraphrases.")

        # Generate token ids
        outputs = self.paraphrasing_model.generate(
            **inputs if isinstance(inputs, transformers.BatchEncoding) else inputs,
            num_return_sequences=num_paraphrases,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            **kwargs
        )

        # Convert token ids to a list of paraphrases
        return self.paraphrasing_tokenizer.batch_decode(
            outputs[:, num_input_tokens],
            skip_special_tokens=True
        )

    def generate_translations(self) -> list[str]:
        ...

    def run(self,
            input_path: Path | str,
            output_path: Path | str,
            augmentation_strategy: AugmentationStrategyLiteral,
            batch_size: int,
            model: Optional[Path | str] = None,
            batch_load_input: bool = False,
            start_line_idx: int = 1,
            **kwargs
            ):
        """Run the whole augmentation pipeline."""

        augmentation_strategy = AugmentationStrategy(augmentation_strategy)
        ...
