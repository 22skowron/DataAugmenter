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
    TextGenerationPipeline,
    Text2TextGenerationPipeline,
    pipeline,
    BatchEncoding
)
import torch

from utils.types import (
    InputJSON,
    ParaphrasedJSON,
    TranslatedJSON,
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
            mode: File mode — `a` to append or `w` to overwrite. Defaults to w.
            encoding: Encoding used to write the file. Defaults to `utf-8`.
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
    def generate(self, *args, **kwargs) -> list[list[str]]:
        """Generate augmented variations of the input text."""
        ...

    @abstractmethod
    def run(
            self,
            input_file: str | Path,
            output_file: str | Path,
            batch_size: int,
            num_augmentations: int,
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
            padding_side="left",
            **kwargs
        )
        self.pipeline: TextGenerationPipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def get_prompt(
            self,
            text: str,
            prompt_type: PromptTypeLiteral = "basic",
            paraphrase_examples: Optional[list[str]] = None,
            custom_prompt: Optional[list[dict[str, str]]] = None
    ) -> list[dict[str, str]]:
        """
        Get chat-like prompt with paraphrasing instructions for the LLM.

        Args:
            text: Original text chunk.
            prompt_type: Type of prompt to use. Available options:
                - basic: Simple instruction to generate paraphrases.
                - strict: Not implemented! Instruction to generate paraphrases with additional instructions e.g. not to generate Markdown symbols and add additional comments.
                - few_shot: Not implemented!
                - custom: Not implemented!
            paraphrase_examples: List of example paraphrases to bake into few_shot prompt. Only used when prompt_type=few_shot.
            custom_prompt: Custom prompt template. Only used when prompt_type=custom.

        Returns:
            List of dictionaries with keys role and content.
        """
        prompt_stack = None

        prompt_type = PromptType(prompt_type)
        if prompt_type == PromptType.BASIC:
            prompt_stack = [
                {"role": "system",
                 "content": "Jesteś asystentem AI, który specjalizuje się w parafrazowaniu tekstu."},
                {"role": "user", "content": f"Proszę sparafrazuj następujący tekst: {text}"}
            ]
        elif prompt_type == PromptType.STRICT:
            raise NotImplementedError
        elif prompt_type == PromptType.FEW_SHOT:
            raise NotImplementedError
        elif prompt_type == PromptType.CUSTOM:
            raise NotImplementedError

        return prompt_stack

    def estimate_max_new_tokens(
            self,
            texts: str | list[str],
            len_factor: float = 1.4
    ) -> int:
        """
        Compute a reasonable estimate of `max_new_tokens` generation parameter for a paraphrasing task.

        Favours conservative approach. When passed a list, the estimate is computed for the longest sequence.

        Args:
            texts: Text(s) for which to estimate `max_new_tokens`. Either a string or a list of strings.
            len_factor: Multiplier used to compute the maximum paraphrase length based on the input length. Defaults to 1.4.

        Returns:
            Estimate of `max_new_tokens`.
        """
        longest_text = texts if isinstance(texts, str) else max(texts, key=len)

        encoded = self.tokenizer(longest_text)
        num_input_tokens = len(encoded["input_ids"])

        return int(num_input_tokens * len_factor)

    def generate(
            self,
            texts: list[str] | list[dict[str, str]],
            num_augmentations: int,
            max_new_tokens: int,
            temperature: float = 0.7,
            **kwargs
    ) -> list[list[str]]:
        """
        Batch generate text augmentations based on provided encodings.

        Args:
            texts: Generation prompts to complete. Either simple strings or dicts with `role` and `content` keys. If a singular value is passed its wrapped in a list.
            num_augmentations: Number of sequences to generate per prompt.
            max_new_tokens: Maximum number of generation tokens.
            temperature: Sampling temperature for generation. Defaults to 0.7.
            **kwargs: Additional keyword arguments passed to transformers' pipeline (and so to model's generation method).

        Returns:
             List of lists with generated augmentations (one list per prompt).
        """
        outputs = self.pipeline(
            text_inputs=texts,
            num_return_sequences=num_augmentations,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_full_text=False,
            **kwargs
        )

        augmentations = []
        for aug_list in outputs:
            gen_texts = [gen_text["generated_text"] for gen_text in aug_list]
            augmentations.append([gen_texts])

        return augmentations

    def run(
            self,
            input_file: str | Path,
            output_file: str | Path,
            batch_size: int,
            num_augmentations: int,
            start_line_idx: int = 1,
            prompt_type: PromptTypeLiteral = "basic",
            len_factor: float = 1.4,
            temperature: float = 0.7,
            **kwargs
    ):
        """

        Args:
            num_augmentations: Number of sequences to generate.
            len_factor: Multiplier used to compute the maximum output length based on the input length. Defaults to 1.4.
            temperature: Sampling temperature for generation. Defaults to 0.7.
            **kwargs: Additional keyword arguments passed to model's generation method.

        Returns:
            List of generated sequences.
        """
        pass
