"""
Abstract base classes and implementations for text data augmentation using Large Language Models.

This module defines the core interfaces and utilities for building text data augmentation
pipelines. It includes general-purpose and LLM-based augmenter base classes, as well as
concrete implementations for a variety of augmentation strategies.
"""

import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Literal, Optional, cast

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextGenerationPipeline,
    Text2TextGenerationPipeline,
    pipeline,
)
import torch

from utils.types import (
    InputJSON,
    ParaphrasedJSON,
    TranslatedJSON,
    ChatTemplate,
    PromptType,
    PromptTypeLiteral
)
from utils.logger import logger


class DataAugmenter(ABC):
    """
    Base class for all data augmenters.

    Defines the standard interface for text data augmentation pipelines.
    This is an abstract class and should not be instantiated directly.
    All custom augmenters should inherit from this class and implement the `run` method.
    """

    @abstractmethod
    def run(
            self,
            input_file: str | Path,
            output_file: str | Path,
            num_augmentations: int,
            start_line_idx: int = 1,
            **kwargs
    ):
        """
        Run the whole augmentation pipeline.

        Wrapper method which handles loading the dataset, generating text augmentations and saving them.
        Expects both the input and output files to be of JSON Lines (JSONL) format.

        Args:
            input_file: The file path from where the dataset will be loaded.
            output_file: The file path where augmentations will be saved.
            num_augmentations: Number of sequences to generate.
            start_line_idx: Index of the first line of the input file which will be processed, 1-based. Defaults to 1.
            **kwargs: Additional keyword arguments passed to downstream methods.
        """
        pass

    @staticmethod
    def load_dataset(
            file: str | Path,
            batch_size: int = 1,
            start_line_idx: int = 1,
            encoding: str | None = "utf-8"
    ) -> Iterator[InputJSON] | Iterator[list[InputJSON]]:
        """
        Load dataset from a JSONL file.

        Returns an Iterator over the dataset. Only the asked for portion of dataset is loaded to memory.
        If `batch_size` = 1 yields the actual JSON objects, otherwise a list of objects.
        Setting `batch_size` > 1 enables batched behaviour.

        Args:
            file: The file path from where the dataset will be loaded.
            batch_size: Number of objects in a batch. Defaults to 1.
            start_line_idx: Index of the first line of the input file which will be processed, 1-based. Defaults to 1.
            encoding: Encoding used to read the file. Defaults to `utf-8`.

        Yields:
            InputJSON object(s): Either a single JSON object or a list of them (batch).
        """
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        start_line_idx -= 1

        with open(file, "r", encoding=encoding) as f:
            batch = []

            for line_idx, line in enumerate(f):
                if line_idx < start_line_idx:
                    continue

                obj = json.loads(line)

                if batch_size == 1:
                    yield obj
                else:
                    batch.append(obj)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []

            # Yield any remaining items in batch
            if len(batch) > 0:
                yield batch

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
    """
    Base class for data augmenters that utilize Large Language Models (LLMs).

    Extends the DataAugmenter interface by adding a `generate` method for producing
    augmentations using LLMs. Designed for scenarios where text generation is required
    as part of the augmentation pipeline.

    This is an abstract class and should not be instantiated directly.
    All LLM-based augmenters should inherit from this class and implement the `generate` and `run` methods.

    Attributes:
        device: The torch device on which the model will run. Defaults to `cuda` (if available) or `cpu`.
    """

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
    def generate(
            self,
            texts: list[str],
            num_augmentations: int,
            max_new_tokens: int,
            temperature: float = 0.7,
            **kwargs
    ) -> list[list[str]]:
        """
        Batch generate text augmentations.

        Args:
            texts: Input sequences for generation.
            num_augmentations: Number of sequences to generate per input sequence.
            max_new_tokens: Maximum number of generation tokens.
            temperature: Sampling temperature for generation. Defaults to 0.7.
            **kwargs: Additional keyword arguments passed to downstream methods.

        Returns:
             List of lists with generated augmentations (one list per input sequence).
        """
        pass

    @abstractmethod
    def run(
            self,
            input_file: str | Path,
            output_file: str | Path,
            num_augmentations: int,
            start_line_idx: int = 1,
            batch_size: int = 8,
            len_factor: float = 1.4,
            temperature: float = 0.7,
            **kwargs
    ):
        """
        Run the whole augmentation pipeline.

        Wrapper method which handles loading the dataset, generating text augmentations and saving them.
        Expects both the input and output files to be of JSON Lines (JSONL) format.

        Args:
            input_file: The file path from where the dataset will be loaded.
            output_file: The file path where augmentations will be saved.
            num_augmentations: Number of sequences to generate.
            start_line_idx: Index of the first line of the input file which will be processed, 1-based. Defaults to 1.
            batch_size: Number of texts to process at a time. Defaults to 8.
            len_factor: Multiplier used to compute the maximum output length based on the input length. Defaults to 1.4.
            temperature: Sampling temperature for generation. Defaults to 0.7.
            **kwargs: Additional keyword arguments passed to downstream methods.
        """
        pass


class ParaphrasingAugmenter(LLMDataAugmenter):
    """
    Data augmenter that generates paraphrases using a pre-trained language model.

    Loads a causal language model and tokenizer to generate paraphrased versions of input text.
    Supports batch processing and configurable prompt styles. The core logic builds on top of
    LLMDataAugmenter to implement a full paraphrasing pipeline.

    Attributes:
        model: The pre-trained causal language model used for generation.
        tokenizer: The tokenizer associated with the model.
        pipeline: A HuggingFace text-generation pipeline for generating paraphrases.
        device: The torch device on which the model will run. Defaults to `cuda` (if available) or `cpu`.
    """

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
    ) -> ChatTemplate:
        """
        Get chat-like prompt with paraphrasing instructions for the LLM.

        Args:
            text: Original text chunk.
            prompt_type: Type of prompt to use. Available options:
                - `basic`: Simple instruction to generate paraphrases.
                - `strict`: Not implemented! Instruction to generate paraphrases with additional instructions e.g. not to generate Markdown symbols and add additional comments.
                - `few_shot`: Not implemented!
                - `custom`: Not implemented!
            paraphrase_examples: List of example paraphrases to bake into `few_shot` prompt. Only used when `prompt_type`=`few_shot`.
            custom_prompt: Custom prompt template. Only used when `prompt_type`=`custom`.

        Returns:
            ChatTemplate with paraphrasing instructions.
        """
        prompt_stack: ChatTemplate = []

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
            texts: list[str] | list[ChatTemplate],
            num_augmentations: int,
            max_new_tokens: int,
            temperature: float = 0.7,
            **kwargs
    ) -> list[list[str]]:
        """
        Batch generate text augmentations.

        Args:
            texts: Generation prompts to complete. Either list of strings or list of ChatTemplates.
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
            augmentations.append(gen_texts)

        return augmentations

    def run(
            self,
            input_file: str | Path,
            output_file: str | Path,
            num_augmentations: int,
            start_line_idx: int = 1,
            batch_size: int = 8,
            prompt_type: PromptTypeLiteral = "basic",
            len_factor: float = 1.4,
            temperature: float = 0.7,
            **kwargs
    ):
        """
        Run the whole paraphrasing pipeline.

        Wrapper method which handles loading the dataset, generating paraphrases and saving them.
        Expects both the input and output files to be of JSON Lines (JSONL) format.

        Args:
            input_file: The file path from where the dataset will be loaded.
            output_file: The file path where augmentations will be saved.
            num_augmentations: Number of sequences to generate.
            batch_size: Number of texts to process at a time. Defaults to 8.
            start_line_idx: Index of the first line of the input file which will be processed, 1-based. Defaults to 1.
            prompt_type: Type of prompt to use. Available options:
                - `basic`: Simple instruction to generate paraphrases.
                - `strict`: Not implemented! Instruction to generate paraphrases with additional instructions e.g. not to generate Markdown symbols and add additional comments.
                - `few_shot`: Not implemented!
                - `custom`: Not implemented!
            len_factor: Multiplier used to compute the maximum output length based on the input length. Defaults to 1.4.
            temperature: Sampling temperature for generation. Defaults to 0.7.
            **kwargs: Additional keyword arguments passed to transformers' pipeline (and so to model's generation method).
        """
        encoding = kwargs.get("encoding", "utf-8")
        paraphrase_examples = kwargs.get("paraphrase_examples", None)
        custom_prompt = kwargs.get("custom_prompt", None)

        dataset = self.load_dataset(
            file=input_file,
            batch_size=batch_size,
            start_line_idx=start_line_idx,
            encoding=encoding
        )

        batch_idx = 1
        try:
            for batch_idx, batch in enumerate(dataset, start=1):
                if batch_size == 1:
                    batch = [batch]
                logger.info(f"📦 Processing batch: {batch_idx} (length: {len(batch)}) ...")

                texts = [obj["text"] for obj in batch]

                chat_prompts = [self.get_prompt(
                    text=obj["text"],
                    prompt_type=prompt_type,
                    paraphrase_examples=paraphrase_examples,
                    custom_prompt=custom_prompt
                ) for obj in batch]

                max_new_tokens = self.estimate_max_new_tokens(
                    texts=texts,
                    len_factor=len_factor
                )

                paraphrases = self.generate(
                    texts=chat_prompts,
                    num_augmentations=num_augmentations,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )

                batch = cast(list[ParaphrasedJSON], batch)
                for i, obj in enumerate(batch):
                    obj["paraphrases"] = paraphrases[i]

                self.save_augmentations(
                    augmentations=batch,
                    file=output_file,
                    mode="a",
                    encoding=encoding
                )

            logger.info("✅ Augmentation completed.")

        except Exception as e:
            logger.exception(f"❗Pipeline failed at batch: {batch_idx}.")

            restart_line_idx = (batch_idx - 1) * batch_size + start_line_idx
            logger.info(
                f"💡 You can restart generation from where the pipeline failed by setting "
                f"`start_line_idx = {restart_line_idx}`."
            )
            raise