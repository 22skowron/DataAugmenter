from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from typing import List, Iterator, Tuple, Literal, Optional, TypedDict

from linear_script import outputs, max_new_tokens, paraphrases
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

    @staticmethod
    def load_dataset(
            dataset_path: str | Path,
            batch_size: int,
            start_line_idx: int = 1,
            encoding: str | None = "utf-8"
    ) -> Iterator[list[InputJSON]]:

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive int.")

        start_line_idx -= 1

        with open(dataset_path, "r", encoding=encoding) as f:
            batch = []
            for line_idx, line in enumerate(f):
                if line_idx < start_line_idx:
                    continue

                batch.append(json.loads(line))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    @staticmethod
    def save_augmentations(
            output_path: str | Path,
            augmentations: list[ParaphrasedJSON] | list[TranslatedJSON],
            mode: str = "w",
            encoding: str | None = "utf-8"
    ):
        with open(output_path, mode, encoding=encoding) as f:
            for obj in augmentations:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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

    def encode_paraphrases(
            self,
            prompts: list[str],
            padding: str | bool = 'longest',
            truncation: str | bool = True,
            **kwargs
    ) -> transformers.BatchEncoding:
        if not self.paraphrasing_tokenizer:
            raise ValueError(
                "Tokenizer of a paraphrasing LLM is not set. Please initialize it before calling encode_paraphrases.")

        return self.paraphrasing_tokenizer(
            prompts,
            return_tensors='pt',
            padding=padding,
            padding_side='left',
            truncation=truncation,
            **kwargs
        ).to(self.device)

    def encode_translations(self) -> transformers.BatchEncoding:
        ...

    def count_prompt_template_tokens(
            self,
            prompt_type: PromptTypeLiteral = "basic",
            paraphrase_examples: Optional[list[str]] = None,
            custom_prompt: Optional[list[dict[str, str]]] = None
    ) -> int:
        empty_prompt_template = self.get_prompt_str(
            og_text="",
            prompt_type=prompt_type,
            paraphrase_examples=paraphrase_examples,
            custom_prompt=custom_prompt
        )

        return self.encode_paraphrases(
            [empty_prompt_template]
        )["input_ids"].shape[1]

    @staticmethod
    def estimate_max_new_tokens(
            num_text_tokens: int,
            augmentation_strategy: AugmentationStrategyLiteral,
            num_prompt_template_tokens: Optional[int] = None,
            len_factor: float = 1.4,
    ) -> int:
        augmentation_strategy = AugmentationStrategy(augmentation_strategy)

        max_new_tokens: int = 0
        if augmentation_strategy == AugmentationStrategy.PARAPHRASING:
            if not isinstance(num_prompt_template_tokens, int):
                raise ValueError(
                    f"Invalid value for `num_prompt_template_tokens`. "
                    f"When augmentation_strategy is set to {AugmentationStrategy.PARAPHRASING.value} `num_prompt_template_tokens` is mandatory with type int."
                )

            num_og_text_tokens = num_text_tokens - num_prompt_template_tokens  # ~Number of InputJSON["text"] tokens of the longest text in a batch
            max_new_tokens = int(
                num_og_text_tokens * len_factor)  # ⚠ model.generate does NOT support per-example max length in batch mode, so here we must take the MAX of the batch for safety (conservative estimate)

        elif augmentation_strategy == AugmentationStrategy.TRANSLATION:
            max_new_tokens = int(num_text_tokens * len_factor)

        return max_new_tokens

    def generate_paraphrases(
            self,
            inputs: transformers.BatchEncoding | torch.Tensor,
            num_paraphrases: int,
            num_input_tokens: int,
            max_new_tokens: int,
            temperature: float = 0.7,
            **kwargs
    ) -> list[str]:
        if self.paraphrasing_model is None or self.paraphrasing_tokenizer is None:
            raise ValueError("Paraphrasing LLM is not set. Please initialize it before calling generate_paraphrases.")

        if (num_input_tokens + max_new_tokens >= self.paraphrasing_tokenizer.model_max_length):
            logger.warning(
                f"⚠️ num_input_tokens + max_new_tokens ({num_input_tokens + max_new_tokens}) >= context window ({self.paraphrasing_tokenizer.model_max_length})"
            )

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
            num_augmentations: int,
            batch_size: int,
            temperature: float = 0.7,
            model: Optional[Path | str] = None,
            start_line_idx: int = 1,
            **kwargs
            ):
        """Run the whole augmentation pipeline."""

        encoding = kwargs.get("encoding", "utf-8")
        prompt_type = kwargs.get("prompt_type", PromptType.BASIC.value)
        paraphrase_examples = kwargs.get("paraphrase_examples", None)
        custom_prompt = kwargs.get("custom_prompt", None)

        data = self.load_dataset(
            dataset_path=input_path,
            batch_size=batch_size,
            start_line_idx=start_line_idx,
            encoding=encoding
        )

        augmentations: list[ParaphrasedJSON] | list[TranslatedJSON] = []

        augmentation_strategy = AugmentationStrategy(augmentation_strategy)
        if augmentation_strategy == AugmentationStrategy.PARAPHRASING:
            if model:
                self.initialize_paraphrasing_model(model)

            num_prompt_template_tokens = self.count_prompt_template_tokens(
                prompt_type=prompt_type,
                paraphrase_examples=paraphrase_examples,
                custom_prompt=custom_prompt
            )

            for batch_idx, batch in enumerate(data, start=1):
                # 1. Convert raw texts to input prompts
                prompts = [self.get_prompt_str(
                    og_text=obj["text"],
                    prompt_type=prompt_type,
                    paraphrase_examples=paraphrase_examples,
                    custom_prompt=custom_prompt
                ) for obj in batch]

                # 2. Tokenize input prompts
                encoded = self.encode_paraphrases(prompts, **kwargs)

                # 3. Generate paraphrases
                max_new_tokens = self.estimate_max_new_tokens(
                    num_text_tokens=encoded["input_ids"].shape[1],
                    augmentation_strategy=augmentation_strategy.value,
                    num_prompt_template_tokens=num_prompt_template_tokens
                )

                paraphrases = self.generate_paraphrases(
                    inputs=encoded,
                    num_paraphrases=num_augmentations,
                    num_input_tokens=encoded["input_ids"].shape[1],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )

                # 4. Save paraphrases
                self.save_augmentations()

        elif augmentation_strategy == AugmentationStrategy.TRANSLATION:
            if model:
                self.initialize_translation_model(model)
